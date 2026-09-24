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