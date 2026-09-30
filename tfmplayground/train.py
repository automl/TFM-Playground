import os
import time
from dataclasses import dataclass

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn
from torch.utils.data import DataLoader

from tfmplayground.augmentation import add_mixed_widening_features, inject_mcar_missingness
from tfmplayground.callbacks import Callback
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.normalization import compute_target_stats_torch, normalize_targets
from tfmplayground.utils import get_default_device


@dataclass
class WideningConfig:
    """Config for TabPFN-Wide-style feature widening during training. add_features_max == 0
    disables it. Ranges follow the paper: sparsity p in [0, 0.05], noise sigma in [0, 1].
    Widening is mixed (continuous via Algorithm 1, categorical via Algorithm 2); max_cats is
    the maximum categorical cardinality Kmax used by the categorical branch.
    """

    add_features_max: int = 0
    add_features_min: int = 0
    sparsity_max: float = 0.05
    noise_max: float = 1.0
    include_original_prob: float = 0.5
    max_cats: int = 20
    prob_no_widening: float = 0.0  # fracción de batches sin widening (tablas estrechas)

def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    ckpt: dict[str, torch.Tensor] = None,
    multi_gpu: bool = False,
    run_name: str = "tfmplayground",
    missing_rate_max: float = 0.0,
    widening: WideningConfig | None = None,
    amp_dtype: torch.dtype | None = None,
    warmup_steps: int = 0,
):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for,
            the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.
        amp_dtype: (torch.dtype, optional) if set (e.g. torch.bfloat16), runs the forward pass under
            torch.autocast with this dtype. Weights and optimizer state stay in fp32. None disables it.
        warmup_steps: (int) number of optimizer steps of linear learning-rate warmup, from ~0 up to lr.
            Counted in optimizer steps, i.e. batches / accumulate_gradients. 0 disables it.        

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    work_dir = "workdir/" + run_name
    os.makedirs(work_dir, exist_ok=True)
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=lr, weight_decay=0.0, warmup_steps=warmup_steps
    )
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, "num_steps must be divisible by accumulate_gradients"

    mean_loss = float("nan")
    try:
        for epoch in range(ckpt["epoch"] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_batches = 0  # batches actually trained on (batches with NaN targets are skipped)
            optimizer.zero_grad()  # do not carry a partial accumulation over from the previous epoch
            for i, full_data in enumerate(prior):
                train_test_split_index = full_data["train_test_split_index"]
                x = full_data["x"].to(device)
                # Feature widening (HDLSS prior): per batch, sample how many features to add and
                # the sparsity/noise, then generate them per dataset (Algorithm 1 for continuous
                # features, Algorithm 2 for categorical). No-op when disabled.
                if (
                    widening is not None
                    and widening.add_features_max > 0
                    and (widening.prob_no_widening == 0 or torch.rand(1).item() >= widening.prob_no_widening)
                ):
                    num_add = int(
                        torch.randint(widening.add_features_min, widening.add_features_max + 1, (1,)).item()
                    )
                    sparsity = torch.rand(1).item() * widening.sparsity_max
                    noise = torch.rand(1).item() * widening.noise_max
                    x = add_mixed_widening_features(
                        x,
                        num_add,
                        sparsity,
                        noise,
                        max_cats=widening.max_cats,
                        include_original_prob=widening.include_original_prob,
                    )
                # Training-time MCAR augmentation: per batch, draw a rate ~ U(0, missing_rate_max)
                # and set feature cells to NaN, so the model learns to use the missing indicator.
                if missing_rate_max > 0:
                    rate = torch.rand(1, device=x.device).item() * missing_rate_max
                    x = inject_mcar_missingness(x, rate)
                data = (x, full_data["y"][:, :train_test_split_index].to(device))
                # Only guard the targets: features with NaNs are handled by the model
                # (normalize_features imputes them and flags them via the indicator channel),
                # so dropping feature-NaN batches would throw away trainable missing signal.
                if torch.isnan(data[1]).any():
                    continue
                targets = full_data["target_y"].to(device)

                if regression_task:
                    y_mean, y_std = compute_target_stats_torch(data[1])
                    y_norm = normalize_targets(data[1], y_mean, y_std)
                    data = (data[0], y_norm)

                # Mixed precision: matmuls/attention in amp_dtype, weights stay fp32. The loss is
                # computed in fp32. bf16 has fp32's range, so no GradScaler is needed.
                with torch.autocast(
                    device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    output = model(data, train_test_split_index=train_test_split_index)
                output = output.float()
                targets = targets[:, train_test_split_index:]
                if regression_task:
                    targets = normalize_targets(targets, y_mean, y_std)
                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                losses = criterion(output, targets)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients
                num_batches += 1

                # Step on trained batches, not on the loop index: a skipped batch must not shift
                # the accumulation groups.
                if num_batches % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / max(num_batches, 1)
            model.eval()
            optimizer.eval()

            training_state = {
                "epoch": epoch,
                "architecture": {
                    "num_layers": int((model.module if multi_gpu else model).num_layers),
                    "embedding_size": int((model.module if multi_gpu else model).embedding_size),
                    "num_attention_heads": int((model.module if multi_gpu else model).num_attention_heads),
                    "mlp_hidden_size": int((model.module if multi_gpu else model).mlp_hidden_size),
                    "num_outputs": int((model.module if multi_gpu else model).num_outputs),
                },
                "model": (model.module if multi_gpu else model).state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(training_state, work_dir + "/latest_checkpoint.pth")

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        (model.module if multi_gpu else model),
                        dist=criterion,
                    )
                else:
                    callback.on_epoch_end(
                        epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model)
                    )
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), mean_loss
