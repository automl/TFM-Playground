import math

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train


class _MockPrior:
    """Minimal prior: yields the given batches and reports its length/num_steps."""

    def __init__(self, batches):
        self._batches = batches
        self.num_steps = len(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _batch(x, y, target_y, tts):
    return {"x": x, "y": y, "target_y": target_y, "train_test_split_index": tts}


def _tiny_model():
    torch.manual_seed(0)
    return NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10
    )


def test_train_processes_batch_with_nan_in_features(tmp_path, monkeypatch):
    """A batch with NaNs only in the features is no longer skipped: normalize_features
    imputes them and the indicator channel flags them, so the batch contributes to training.
    """
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    x = torch.randn(1, 6, 3)
    x[0, 0, 0] = float("nan")                 # NaN in a feature
    y = torch.randint(0, 10, (1, 6)).float()
    prior = _MockPrior([_batch(x, y, y.clone(), tts=4)])

    _, total_loss = train(
        model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run"
    )

    assert total_loss != 0.0                  # the batch was processed, not skipped


def test_train_skips_batch_with_nan_in_targets(tmp_path, monkeypatch):
    """A batch with NaNs in the training targets is still skipped, because pad_targets has
    no NaN handling and a missing label has no meaning.
    """
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    x = torch.randn(1, 6, 3)
    y = torch.randint(0, 10, (1, 6)).float()
    y[0, 0] = float("nan")                    # NaN in a training target
    prior = _MockPrior([_batch(x, y, y.clone(), tts=4)])

    _, total_loss = train(
        model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run"
    )

    assert total_loss == 0.0                  # the batch was skipped


def test_train_injects_missingness_when_rate_positive(tmp_path, monkeypatch):
    """With missing_rate_max > 0, features get NaNs injected and the batch still trains
    (the model handles them), exercising the indicator channel end-to-end.
    """
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    x = torch.randn(1, 8, 3)                     # no NaN from the prior
    y = torch.randint(0, 10, (1, 8)).float()
    prior = _MockPrior([_batch(x, y, y.clone(), tts=5)])

    _, total_loss = train(
        model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"),
        run_name="run", missing_rate_max=0.5,
    )

    assert total_loss != 0.0                     # trained with injected missingness, no crash


def test_train_widens_features_when_configured(tmp_path, monkeypatch):
    """With a WideningConfig whose add_features_max > 0, the batch is widened (HDLSS prior)
    and still trains end-to-end.
    """
    from tfmplayground.train import WideningConfig

    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    x = torch.randn(1, 8, 3)                     # 3 features from the prior
    y = torch.randint(0, 10, (1, 8)).float()
    prior = _MockPrior([_batch(x, y, y.clone(), tts=5)])

    _, total_loss = train(
        model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"), run_name="run",
        widening=WideningConfig(add_features_min=10, add_features_max=10, sparsity_max=0.1, noise_max=0.5),
    )

    assert total_loss != 0.0                     # trained on widened features, no crash


def test_train_runs_with_bf16_autocast(tmp_path, monkeypatch):
    """With amp_dtype=torch.bfloat16 the forward runs under autocast, the loss is finite and
    the weights stay in fp32 (autocast never changes parameter dtypes).
    """
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    x = torch.randn(1, 8, 3)
    y = torch.randint(0, 10, (1, 8)).float()
    prior = _MockPrior([_batch(x, y, y.clone(), tts=5)])

    trained, total_loss = train(
        model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"),
        run_name="run", amp_dtype=torch.bfloat16,
    )

    assert math.isfinite(total_loss) and total_loss > 0
    assert all(p.dtype == torch.float32 for p in trained.parameters())


def test_train_warmup_shrinks_first_update(tmp_path, monkeypatch):
    """With a long warmup the first optimizer step uses a tiny fraction of lr, so the weights
    move much less than without warmup, starting from the same initialization.
    """
    monkeypatch.chdir(tmp_path)
    x = torch.randn(1, 8, 3)
    y = torch.randint(0, 10, (1, 8)).float()

    def weight_change(warmup_steps):
        model = _tiny_model()                          # same seed -> same init every call
        before = [p.detach().clone() for p in model.parameters()]
        prior = _MockPrior([_batch(x, y, y.clone(), tts=5)])
        trained, _ = train(
            model, prior, nn.CrossEntropyLoss(), epochs=1, device=torch.device("cpu"),
            run_name="run", warmup_steps=warmup_steps,
        )
        return sum((p.detach() - b).abs().sum().item() for p, b in zip(trained.parameters(), before))

    assert weight_change(warmup_steps=1000) < weight_change(warmup_steps=0)