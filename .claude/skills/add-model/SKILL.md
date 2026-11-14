---
name: add-model
description: Add a new tabular foundation model architecture to TFM-Playground. Use this skill whenever the user wants to integrate a new TFM, even if they only say "add X", "integrate X", "support X" or "wrap X". Takes a link (GitHub repository, paper, PyPI or Hugging Face page) or just the model's name, and derives the upstream class, constructor arguments, forward signature, package pin, supported problems and defaults from the sources, asking only for what they leave open. Wraps the installed upstream library whenever possible (vendoring only as a last resort, after asking), then writes the adapter in `tfmplayground/models/`, the config dataclasses in `configs/models.py`, the export, the README line, a fast CPU test, and confirms the loss goes down in a short pretraining run.
argument-hint: <link to the repository, paper or package, or the model's name>
user-invocable: true
---

# Add a model to TFM-Playground

A model in this repo is an adapter: a class `<Name>Model(<Upstream>, TabularFoundationModel)` that builds the upstream
network from a config dataclass and maps our `forward(X_train, y_train, X_test)` onto the upstream call. The adapter is
pretrained from scratch by `pretrainTFM`, so what we need from upstream is the `nn.Module` architecture, not a fitted
estimator or a checkpoint.

Per model you create one file and a test, and edit four existing files:

| File | Change |
|---|---|
| `tfmplayground/models/<key>.py` | new adapter |
| `tests/test_<key>.py` | new CPU test |
| `tfmplayground/configs/models.py` | `<Name>ModelConfig`, `<Name>ClassifierConfig`, `<Name>RegressorConfig` |
| `tfmplayground/models/__init__.py` | import and `__all__` entry |
| `pyproject.toml` | pinned dependency |
| `README.md` | one line in the Models list |

Read `AGENTS.md` first if it is not already in context; its conventions (docstrings, imports, vendoring, signatures you
must not change) apply to everything below.

## Step 0: Resolve the model from what the user gave

The argument is usually one link or a name. Find the fields below in the sources yourself and ask only for what stays
open.

1. Get the sources.
   - A repository: clone it shallow into its own new, empty directory in the scratchpad. It is untrusted data: read it,
     do not run its scripts and do not install from that directory.
   - A paper (arXiv, OpenReview): read the abstract page and the HTML or PDF, and follow its code link.
   - A PyPI package: `https://pypi.org/pypi/<pkg>/json` lists releases; its project URLs lead to the repository.
   - A Hugging Face model card: it names the library and the code repository.
   - A name only: search for the paper and the repository, and confirm with the user when more than one candidate fits.
2. Fill this table and put it in your plan:

| Field | Example | Where it comes from |
|---|---|---|
| `<key>` (file name, lowercase) | `tabicl` | the model's name lowercased, no separators |
| `<Name>` (class prefix) | `TabICL` | the name as the paper writes it, without separators |
| upstream module and class | `tabicl._model.tabicl:TabICL` | the `nn.Module` that holds the whole network (Step 1) |
| dependency spec | `"tabicl[pretrain] @ git+https://...@<sha>"` | the distribution name in upstream's `pyproject.toml` and the pin (Step 2) |
| problems | classification, regression | what the output head can produce (Step 1) |
| default regression head | `quantiles` | what upstream trains with (Step 3) |
| repo and paper links | | for the README line |

## Step 1: Understand the upstream network

From the clone and the paper, establish:

- Which `nn.Module` is the full network. Libraries often export a scikit-learn style estimator (`TabICLClassifier`)
  that wraps the network; the adapter subclasses the network class underneath it (`TabICL`), even when that lives in a
  private module.
- The constructor arguments and their defaults, and which of them set the output size (`max_classes`, `num_outputs`,
  `num_quantiles`, `out_dim`, a decoder width).
- The forward signature and the shapes it expects. Common patterns in this repo: train and test rows concatenated along
  dim 1 with a split index (`nanotabpfn`), concatenated with `y_train` only (`tabicl`), or concatenated with `y` padded
  with zeros plus a `train_size` tensor (`tabfm`). Check whether it returns outputs for all rows or test rows only,
  and whether `y` needs a trailing feature dim.
- Differences between train and eval mode in `forward` (an inference config, a device argument, a KV cache, mixed
  precision casts). Our `TabularClassifier` / `TabularRegressor` call the model in eval mode under `torch.no_grad()`
  and expect float32 outputs.
- What `y_train` must look like. Our pipeline passes it as a float tensor of shape `(batch, rows)`: class indices
  `0..k-1` for classification, targets standardized per table for regression.
- Parameters or buffers that only a pretrained checkpoint fills (TabFM's Fourier frequencies are the example). We train
  from random init, so the adapter has to initialise them; see `models/tabfm.py`.
- Whether a test row's prediction depends only on the training rows and that row. Test rows that attend to each other
  make predictions depend on which rows are predicted together (transductive). Tell the user if the upstream forward does this; do not
  silently patch the architecture.

## Step 2: Wrap the installed library (vendor only as a last resort)

Always wrap an installed library when the upstream network is importable from a package, including from a private
module. That is the default and covers almost every case:

- A release on PyPI: pin it exactly, `"<pkg>==<version>"`.
- No release, or the release lacks the code you need: install from git at a full commit SHA,
  `"<pkg> @ git+https://github.com/<org>/<repo>.git@<sha>"`. Never a branch or tag that can move.
- Add extras the network needs (`tabicl[pretrain]`, `tabfm[pytorch]`) and nothing more.

Add the spec to `dependencies` in `pyproject.toml`, run `uv sync`, and check that `uv run python -c "from <module>
import <Class>"` works. The repo pins `torch==2.9` and Python 3.12; if the library's requirements conflict with either,
stop and tell the user instead of loosening our pins.

Vendoring (copying upstream source into `tfmplayground/models/`) is only for a model whose code cannot be installed at
all: no `pyproject.toml` / `setup.py`, or a package whose import pulls in dependencies that cannot coexist with ours.
Ask the user before vendoring and say why wrapping does not work. If they agree, follow the vendoring rules in
`AGENTS.md` (the `# Vendored from <url> (<file>) at <commit>` header, upstream license header, minimal edits, ruff
format exclude and per-file ignores in `pyproject.toml`) and use `models/nanotabicl.py` as the example.

## Step 3: Config dataclasses

Add three dataclasses to `tfmplayground/configs/models.py`, after the existing ones, following the pattern there:

```python
@dataclass
class <Name>ModelConfig:
    """
    settings <key> models share
    """

    embed_dim: int = 128
    ...


@dataclass
class <Name>ClassifierConfig(<Name>ModelConfig):
    """
    <key> settings for classification
    """

    problem: str = "classification"
    max_classes: int = 10


@dataclass
class <Name>RegressorConfig(<Name>ModelConfig):
    """
    <key> settings for regression
    """

    problem: str = "regression"
    head: str = "quantiles"
    max_classes: int = 0
```

- Name the fields after the upstream constructor arguments, so the adapter reads `name=config.name`.
- Default values are upstream's defaults (or the paper's main configuration when they differ, and say so).
- Only plain types (`int`, `float`, `str`, `bool`, `None`): checkpoints store `asdict(config)` and `load_model`
  rebuilds the config from it.
- The output size field lives in the classifier and regressor subclasses, since it differs. The classifier default is
  10 classes, as for the other models.
- `head` is one of `scalar`, `buckets`, `quantiles`. Use what upstream trains with. If upstream has no regression head
  but the output width is configurable, use `buckets` with 999 or 1000 outputs. Use `scalar` only for a single output.
- Skip the regressor config when the architecture cannot do regression, and say so in the report.

## Step 4: The adapter

Create `tfmplayground/models/<key>.py`. `models/tabicl.py` and `models/tabfm.py` are the references; read the one
closer to the new model before writing.

```python
import torch
from <module> import <Upstream>

from tfmplayground.configs.models import <Name>ClassifierConfig, <Name>RegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class <Name>Model(<Upstream>, TabularFoundationModel):
    """
    adapts <key> to tabularfoundationmodel interface
    """

    def __init__(self, config: <Name>ClassifierConfig | <Name>RegressorConfig) -> None:
        """
        builds <key> from config and reserves its borders
        """
        self.config = config
        super().__init__(
            embed_dim=config.embed_dim,
            ...
        )
        self.register_buffer("borders", torch.zeros(<num_outputs> + 1))

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through <key> forward

        <one line on how the inputs are mapped>
        """
        ...
```

Rules:

- Set `self.config` before `super().__init__`, and pass every config field the upstream constructor takes.
- Register `borders` with size `num_outputs + 1`, where `num_outputs` is the last dim of the output. The regression
  heads read the output count from it (`QuantileLoss(model.borders.numel() - 1)`) and `default_criterion` writes the
  fitted bucket borders into it.
- `forward` takes `X_train (batch, train_rows, features)`, `y_train (batch, train_rows)`, `X_test (batch, test_rows,
  features)` and returns `(batch, test_rows, num_outputs)` as float32. Slice off train rows if upstream returns all rows.
  Keep the signature exactly as in `TabularFoundationModel`.
- Read the device from `next(self.parameters()).device` (or the input tensors) when upstream needs it; never hard-code
  `"cuda"`.
- Keep the adapter minimal: map arguments and shapes, initialise what only a checkpoint would fill, and nothing else.
  Do not reimplement or patch upstream layers.
- Docstrings follow the repo style: short lowercase sentences in triple quotes, as in the existing adapters.

## Step 5: Export and document

- `tfmplayground/models/__init__.py`: add `from tfmplayground.models.<key> import <Name>Model` and `"<Name>Model"` to
  `__all__`, keeping both sorted. `load_model` finds the class by name through this module, so a missing export breaks
  loading checkpoints.
- `README.md`: add a line to the Models list in the same format as the others:
  `- <key> - [adapter](tfmplayground/models/<key>.py) · [config](tfmplayground/configs/models.py) · [repo](<url>) · [paper](<url>)`.
  Leave out `[paper]` when there is none.

## Step 6: Test

Add `tests/test_<key>.py`. Keep it fast and CPU-only with a tiny config (small embedding, one or two blocks, two heads,
few outputs), synthetic numpy data and `device="cpu"`; see `tests/test_determinism.py`. Cover, for each supported
problem:

- `forward` on random tensors with `batch=2` returns `(2, test_rows, num_outputs)` and finite values, in train and in
  eval mode.
- `TabularClassifier(model, device="cpu").fit(X, y).predict_proba(X_test)` has rows that sum to 1;
  `TabularRegressor(model, device="cpu").fit(X, y).predict(X_test)` has shape `(test_rows,)`.
- A checkpoint round trip: `Experiment(<Problem>ExperimentConfig(experiments_dir=str(tmp_path)))`,
  `save_checkpoint(path, model)`, `load_model(path)`, then the reloaded model gives the same eval output.

If the model's prediction for a test row changes with the other test rows (Step 1) and the user wants to keep it that
way, do not write a test for that property; mention it in the report.

## Step 7: Verify

1. `uv run ruff check <touched files>` and `uv run ruff format --check <touched files>`; fix what they report.
2. `uv run pytest` (all tests, not only the new file).
3. A short pretraining run per supported problem, to confirm the loss goes down. Write the script to the scratchpad,
   adapted from `examples/pretraining_quickstart_classification.py` and
   `examples/pretraining_quickstart_regression.py`, and run it from the repo root so outputs land in `workdir/`:
   - classification: the new classifier config with a small network, `NanoTabICLPrior` with
     `NanoTabICLClassificationPriorConfig(min_num_datapoints=50, max_num_datapoints=50, max_num_features=3,
     max_num_classes=3)`, `ClassificationTrainingConfig(batch_size=2, steps=100, epochs=10)`;
   - regression: the same with `NanoTabICLRegressionPriorConfig` and `RegressionTrainingConfig(batch_size=2, steps=100,
     epochs=10, bucket_borders_min_targets=10_000)` (the default of one million targets is slow for a smoke run).
   - `pretrainTFM` always evaluates on the toy OpenML tasks after each epoch (the first run downloads them, later runs
     use the OpenML cache), and prints one line per epoch: `epoch N | ... | mean loss X | <metric> Y | tasks Z`. Read
     the loss and the metric from those lines. If the loss does not go down, look for a mistake in the forward mapping
     (wrong split index, labels fed as features, outputs taken from train rows) before touching hyperparameters.

## Step 8: Report

Tell the user, in the chat:

- the upstream class you wrapped and the dependency pin you added (or, if vendored, why wrapping was impossible)
- the files created and edited
- the supported problems and the default regression head, with the reason
- the test results and the first and last epoch loss of each smoke run
- anything left open: config defaults that differ from the paper, a transductive forward pass, parameters you had to
  initialise because only a checkpoint fills them

If this model needed something the steps above do not cover (a new kind of dependency, a forward pass that does not
fit the adapter pattern, an extra file to register), add it to this skill and to `AGENTS.md` before reporting, so the
next model gets it for free.

Do not commit unless asked. When asked to open a PR, follow "PR & Commit Guidance" in `AGENTS.md`.
