import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from typing import Dict
from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree
import os

torch.set_float32_matmul_precision('high')

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device
from gtfm.trainer.muon import Muon



def train(model: NanoTabPFNModel, prior: DataLoader, criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
          epochs: int, accumulate_gradients: int = 1, lr: float = 1e-4, device: torch.device = None,
          callbacks: list[Callback] = None, ckpt: Dict[str, torch.Tensor] = None, multi_gpu: bool = False,
          run_name: str = 'nanoTFM', workdir: str = '.', use_muon: bool = True):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    work_dir = os.path.join(workdir, run_name)
    os.makedirs(work_dir, exist_ok=True)
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)

    if use_muon:
        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if p.ndim != 2:
                adam_params.append(p)
            elif 'transformer_encoder' in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        optimizer_muon = Muon(muon_params, lr=0.1 * lr, momentum=0.95)
        optimizer_adam = schedulefree.AdamWScheduleFree(adam_params, lr=lr, weight_decay=0.0, warmup_steps=1000)
        optimizers = [optimizer_muon, optimizer_adam]
    else:
        optimizer_muon = None
        optimizer_adam = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0, warmup_steps=1000)
        optimizers = [optimizer_adam]

    if ckpt:
        if 'optimizer_adam' in ckpt:
            optimizer_adam.load_state_dict(ckpt['optimizer_adam'])
        elif 'optimizer' in ckpt:  # backwards compat
            optimizer_adam.load_state_dict(ckpt['optimizer'])
        if use_muon and optimizer_muon is not None and 'optimizer_muon' in ckpt:
            optimizer_muon.load_state_dict(ckpt['optimizer_muon'])

    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'

    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer_adam.train()
            total_loss = 0.
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']
                data = (
                    full_data['x'].to(device),
                    full_data['y'][:, :single_eval_pos].to(device),
                    full_data['adj'].to(device),
                )
                if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
                    raise NotImplemented("Should inspect if this is happening")
                targets = full_data['target_y'].to(device)

                if regression_task:
                    y_mean = data[1].mean(dim=1, keepdim=True)
                    y_std = data[1].std(dim=1, keepdim=True) + 1e-8
                    y_norm = (data[1] - y_mean) / y_std
                    data = (data[0], y_norm, data[2])

                device_type = torch.device(device).type
                use_amp = device_type in ("cuda", "mps")
                autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_amp else torch.nullcontext()
                with autocast_ctx:
                    output = model(data, single_eval_pos=single_eval_pos)
                    targets = targets[:, single_eval_pos:]
                    if regression_task:
                        targets = (targets - y_mean) / y_std
                    if classification_task:
                        targets = targets.reshape((-1,)).to(torch.long)
                        output = output.view(-1, output.shape[-1])

                    losses = criterion(output, targets)
                loss = losses.mean() / accumulate_gradients
                if torch.isnan(loss):
                    print('Loss is NaN, stopping training batch.')
                    return full_data
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients
                del output, targets, losses, loss, data, full_data

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    for opt in optimizers:
                        opt.step()
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)

            end_time = time.time()
            mean_loss: float = total_loss / len(prior)
            model.eval()
            optimizer_adam.eval()

            training_state = {
                'epoch': epoch,
                'architecture': {
                    'num_layers': int((model.module if multi_gpu else model).num_layers),
                    'embedding_size': int((model.module if multi_gpu else model).embedding_size),
                    'num_attention_heads': int((model.module if multi_gpu else model).num_attention_heads),
                    'mlp_hidden_size': int((model.module if multi_gpu else model).mlp_hidden_size),
                    'num_outputs': int((model.module if multi_gpu else model).num_outputs)
                },
                'model': (model.module if multi_gpu else model).state_dict(),
                'optimizer_adam': optimizer_adam.state_dict(),
                **(({'optimizer_muon': optimizer_muon.state_dict()} if optimizer_muon is not None else {}))
            }
            torch.save(training_state, work_dir+'/latest_checkpoint.pth')

            for callback in callbacks:
                tabearena_light = False if (epoch == epochs) else True
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model), dist=criterion, tabarena_light=tabearena_light)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model), tabarena_light=tabearena_light)
        callback.on_train_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model), dist=criterion if type(criterion) is FullSupportBarDistribution else None, tabarena_light=False)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss
