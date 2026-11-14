# AGENTS.md

Guidance for coding agents working in this repository. The Documentation for humans is in `README.md`.

## Project Overview

TFM-Playground is a Playground for Tabular Foundation Models (TFMs). It unifies the inference and pretraining interfaces of several TFM architectures and synthetic-data priors, thus enabling easier research on TFMs. It also comes with a compatible pretraining loop and code for periodic evaluation on OpenML tasks, as well as scikit-learn style inference wrappers. The README lists every model and prior with links to its adapter, config, upstream repo and paper.

## Keeping This File and the Skills Up to Date

This file and the skills in `.claude/skills/` (currently only `add-model`) describe the repo as it is. They go stale whenever the repo changes underneath them, and nobody else will notice, so updating them is part of every task:

- When a change affects anything written here (layout, setup or lint commands, CI, testing rules, architecture, conventions, things to avoid), edit the matching section of `AGENTS.md` in the same change. Typical triggers are adding CI, a new test helper or test convention, a new dependency or extra, a new public entry point, or a new subpackage.
- When a change affects what a new model needs (a new required field on model configs, a new base-class method, a new file to register it in, a new test every model must have, downloadable pretrained checkpoints, a new README table), update `.claude/skills/add-model/SKILL.md` so a model added through the skill comes out complete. Apply the same rule to any skill added later.
- Before finishing a task, reread the sections you touched and fix statements that are no longer true, for example "There is no CI in this repo" once CI exists. Remove outdated text instead of adding a correction next to it.
- If the change introduces a recurring workflow similar to adding a model (adding a prior, adding a pretrained checkpoint), mention to the user that a new skill could cover it, but do not create one without asking.

## Repository Layout

A single Python package, `tfmplayground`, plus examples and tests:

- `tfmplayground/models/`: one adapter file per model (`nanotabpfn.py`, `moddednanotabpfn.py`, `nanotabicl.py`, `tabicl.py`, `tabfm.py`). `base.py` holds the `TabularFoundationModel` base class. nanotabpfn, moddednanotabpfn and nanotabicl are the only models where the full code was copied over into this repository, the other files only contain minimal code necessary in order to make the model code from the dependency compatible with our interface.
- `tfmplayground/priors/`: one adapter file per prior. `base.py` holds the `Prior` base class and `PriorDataLoader`. `dump.py` reads and writes HDF5 prior dumps. `main.py` / `__main__.py` is the dump CLI (`python -m tfmplayground.priors`).
- `tfmplayground/configs/`: dataclass configs, one file per area (`models.py`, `priors.py`, `training.py`, `evaluation.py`).
- `tfmplayground/training/`: `pretrain.py` (`pretrainTFM`, the public entry point), `train.py` (the training loop), `callbacks.py` (logging and evaluation callbacks).
- `tfmplayground/evaluation/`: OpenML task lists (`TOY_TASKS`, `TABARENA_TASKS`) and `get_openml_predictions`.
- `tfmplayground/interface.py`: `TabularClassifier` / `TabularRegressor`, the scikit-learn style wrappers around a trained model.
- `tfmplayground/utils.py`: device and seed helpers, regression heads and losses (`BarDistribution`, `FullSupportBarDistribution`, `QuantileLoss`, `ScalarMSELoss`), bucket borders, `Experiment` and `load_model`.
- `examples/`: runnable pretraining scripts (`pretraining_quickstart_classification.py`, `pretraining_quickstart_regression.py`, `pretraining_classification.py`, `pretraining_regression.py`).
- `tests/`: all tests.

## Setup Commands

Requires Python 3.12 exactly (`requires-python = "==3.12.*"`) and [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                                            # Install the package, its dependencies and the dev group (ruff, pytest)
uv sync --extra tensorboard                        # + TensorBoard logging
uv run python examples/pretraining_quickstart_classification.py   # Smoke run, about 5 minutes on a laptop
```

Several dependencies (`tabicl`, `ticl`, `tabpfn-v1-prior`) are installed from git at a pinned commit in `pyproject.toml`. When an adapter breaks after an upstream change, bump the pinned commit deliberately; do not switch to an unpinned branch. `uv.lock` is in `.gitignore` and is not committed.

## Lint & Format

```bash
uv run ruff check .           # Lint
uv run ruff format .          # Format
uv run ruff format --check .  # Check formatting without writing
```

Config lives in `pyproject.toml` (`[tool.ruff]`): line length 120, target `py312`, rules `E`, `F`, `I`, `UP`, `B`, `SIM`. The vendored files (`models/moddednanotabpfn.py`, `models/nanotabicl.py`, `priors/nanotabicl.py`) are excluded from the formatter and have per-file lint ignores, so their diffs against upstream stay small.

There is no CI in this repo. Run `ruff check` and `ruff format --check` on touched files before finishing a task.

## Testing

```bash
uv run pytest                                  # All tests
uv run pytest tests/test_determinism.py        # Single file
uv run pytest -k test_name -x                  # Single test, stop on failure
```

`pyproject.toml` sets `testpaths = ["tests"]`. Keep tests fast and CPU-only: build models with tiny configs (see `tests/test_determinism.py`), pass `device="cpu"`, and use synthetic numpy data instead of OpenML downloads.

## Architecture

### Core data flow

```
Prior.batch() → train() → model(X_train, y_train, X_test) → criterion → checkpoints + evaluation callback → TabularClassifier / TabularRegressor
```

`pretrainTFM(problem, model, prior, eval, training, experiment, device)` in `training/pretrain.py` is the core function wiring everything together. It checks that every config agrees on `problem`, seeds randomness, picks the loss, opens an `Experiment`, logs all configs and then calls `train()`.

### Key abstractions

- **`TabularFoundationModel`** (`models/base.py`): an `nn.Module` whose `forward(X_train, y_train, X_test)` returns predictions for the test rows. `X_train` and `X_test ` and  are `(batch, rows, features)`, `y_train` is `(batch, rows)`, the output is `(batch, test_rows, num_outputs)`. Every adapter subclasses the upstream model and this class, stores `self.config`, and registers a `borders` buffer of size `num_outputs + 1` (used by the regression heads).
- **`Prior`** (`priors/base.py`): `batch(batch_size)` returns `(x_train, y_train, x_test, y_test)` on the prior's device. `PriorDataLoader` turns a prior into an endless iterator. `train()` skips batches with non-finite values, since some prior may occasionally return one.
- **Configs** (`configs/`): plain dataclasses. Each model, prior, training and experiment config comes as a shared base plus a `...Classifier...` / `...Classification...` and a `...Regressor...` / `...Regression...` subclass that sets the `problem` field. `check_problems` in `pretrain.py` rejects a mix of classification and regression configs.
- **Regression heads** (`utils.py`): `model.config.head` is one of `scalar`, `buckets`, `quantiles`. `make_regression_decoder` maps it to `ScalarMSELoss`, `FullSupportBarDistribution` or `QuantileLoss`. For `buckets`, `default_criterion` fits the bucket borders from the prior (`make_bucket_borders`) and writes them into `model.borders` before training. Regression targets are standardized per table inside `train()`.
- **`train()`** (`training/train.py`): currently `schedulefree.AdamWScheduleFree`, gradient clipping, a fixed number of valid steps per epoch, callbacks at the end of each epoch, checkpoints after each epoch. `KeyboardInterrupt` ends training and still closes the callbacks and returns the model.
- **Callbacks** (`training/callbacks.py`): `ClassifierExperimentEvaluationCallback` (mean ROC AUC) and `RegressorExperimentEvaluationCallback` (mean R²) evaluate on the OpenML tasks named by `EvaluationConfig.tasks` (`"toy"`, `"tabarena"`, a list of task ids, or an OpenML suite name) and set `experiment.score`. Console, TensorBoard and Weights & Biases loggers live in the same file.
- **`Experiment`** (`utils.py`): one run directory at `<experiments_dir>/<problem>/<name>/<id>/` (default `workdir/experiments/...`) with a log file and `-ckpt-last.pth` / `-ckpt-best.pth`. Checkpoints store the model and config class names, so `load_model(path)` can rebuild the model with `getattr(models, ...)` and `getattr(configs.models, ...)`.
- **`TabularClassifier` / `TabularRegressor`** (`interface.py`): `fit` stores the preprocessed training data (imputation, ordinal encoding, constant-column removal, standard scaling), `predict` / `predict_proba` run one forward pass with the training data as context. They must stay picklable (TabArena runs them), which is why the preprocessing helpers are module-level functions and not lambdas.

### Prior dumps

Some priors are only available as dumps. `python -m tfmplayground.priors --lib {ticl,tabicl,tabpfn} ...` samples batches through the `...PriorDataLoader` classes in `priors/ticl.py`, `priors/tabpfn.py` and `priors/tabicl.py` and writes them to HDF5 with `dump_prior_to_h5`. `DumpPrior` (`priors/dump.py`) reads a dump back as a `Prior`. It requires the batch size to divide the dump's block size and every table in a batch to share one train/test split. `*.h5` files are gitignored.

## Conventions

- **Add a model**: add `tfmplayground/models/<name>.py` with a class `<Name>Model(<Upstream>, TabularFoundationModel)` that stores `config`, registers the `borders` buffer, and maps `forward(X_train, y_train, X_test)` onto the upstream call. Add `<Name>ModelConfig`, `<Name>ClassifierConfig` and (if supported) `<Name>RegressorConfig` to `configs/models.py`; regressor configs set `head`. Export the model in `models/__init__.py` (`load_model` resolves it by class name) and add a line to the Models list in `README.md`. Use `models/tabicl.py` as the reference for wrapping an installed library and `models/nanotabpfn.py` for code that lives in this repo. The `add-model` skill (`.claude/skills/add-model/SKILL.md`) walks through all of this from a link or a model name; use it when asked to add a model. Wrap an installed library whenever possible and vendor only when the code cannot be installed, after asking the user.
- **Add a prior**: add `tfmplayground/priors/<name>.py` with a `Prior` subclass taking `config` and an optional `device`, add the config classes to `configs/priors.py`, export it in `priors/__init__.py`, and add a line to the Priors list in `README.md`. `priors/tabicl.py` (`TabICLPrior`) is the reference for wrapping a library.
- **Vendored code**: a file copied from another repo starts with `# Vendored from <url> (<file>) at <commit>` plus the upstream license header if it has one. Keep edits to vendored files minimal and add them to the ruff format exclude and per-file ignores instead of reformatting.
- **Docstrings**: short lowercase sentences in triple quotes (`"""gives training config that matches problem"""` style, as in `pretrain.py` and `configs/`). Public entry points (`pretrainTFM`, `train`) add numpydoc `Parameters` / `Returns` sections. Match the file you are editing.
- **Imports**: absolute imports rooted at `tfmplayground` (e.g. `from tfmplayground.models.base import TabularFoundationModel`). `priors/__init__.py` and `priors/main.py` use relative imports; leave them as they are.
- **Problems**: the only problem values are `"classification"` and `"regression"`. A new config that knows its problem gets a `problem` field so `check_problems` can see it.
- **Outputs**: runs write to `workdir/`, which is gitignored. Do not write checkpoints, dumps or logs anywhere else in the repo.
- **No new top-level docs files** unless the user asks. Edit existing files in place.
- **Prose style**: everything you write for a human reader (docstrings, comments, markdown, commit messages, PR descriptions, chat replies) follows [AI Writing Tropes to Avoid](#ai-writing-tropes-to-avoid) at the bottom of this file.

## PR & Commit Guidance

- Keep commits focused; do not bundle unrelated refactors with bug fixes.
- Run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` before opening a PR. There is no CI to catch it afterwards.
- Check that `AGENTS.md` and `.claude/skills/add-model/SKILL.md` still match the code after your change (see [Keeping This File and the Skills Up to Date](#keeping-this-file-and-the-skills-up-to-date)) and include any updates in the same PR.
- For a change to the training loop, a model or a prior, also run `uv run python examples/pretraining_quickstart_classification.py` or `examples/pretraining_quickstart_regression.py` (or a shortened copy of it) and confirm the loss goes down.
- PR description: a two-to-four sentence summary first, then a `<details><summary>Details</summary>` block (collapsed by default) with the longer description. Keep the whole body short.

## Things to Avoid

- Do not change the signature of `TabularFoundationModel.forward`, `Prior.batch`, `pretrainTFM`, `TabularClassifier` or `TabularRegressor` without explicit user direction; the examples, the README and downstream users depend on them.
- Do not rename model or config classes casually: checkpoints store their class names and `load_model` looks them up by name, so a rename breaks every existing checkpoint.
- Do not use lambdas or local functions inside `TabularClassifier` / `TabularRegressor` state; it breaks pickling.
- Do not reformat vendored files or unpin the git dependencies in `pyproject.toml`.
- Do not add slow, GPU-only or network-dependent tests to `tests/`.

---

# AI Writing Tropes to Avoid

Applies to everything you write in this repo that a human reads: docstrings,
comments, markdown docs, commit messages, PR descriptions, user-facing copy,
and your replies in the chat.

Source: [tropes.fyi](https://tropes.fyi) by [ossama.is](https://ossama.is)

---

## Word Choice

### "Quietly" and Other Magic Adverbs

Overuse of "quietly" and similar adverbs to convey subtle importance or understated power. AI reaches for these adverbs to make mundane descriptions feel significant. Also includes: "deeply", "fundamentally", "remarkably", "arguably".

**Avoid patterns like:**
- "quietly orchestrating workflows, decisions, and interactions"
- "the one that quietly suffocates everything else"
- "a quiet intelligence behind it"

### "Delve" and Friends

Used to be the most infamous AI tell. "Delve" went from an uncommon English word to appearing in a staggering percentage of AI-generated text. Part of a family of overused AI vocabulary including "certainly", "utilize", "leverage" (as a verb), "robust", "streamline", and "harness".

**Avoid patterns like:**
- "Let's delve into the details..."
- "Delving deeper into this topic..."
- "We certainly need to leverage these robust frameworks..."

### "Tapestry" and "Landscape"

Overuse of ornate or grandiose nouns where simpler words would do. "Tapestry" is used to describe anything interconnected. "Landscape" is used to describe any field or domain. Other offenders: "paradigm", "synergy", "ecosystem", "framework".

**Avoid patterns like:**
- "The rich tapestry of human experience..."
- "Navigating the complex landscape of modern AI..."
- "The ever-evolving landscape of technology..."

### The "Serves As" Dodge

Replacing simple "is" or "are" with pompous alternatives like "serves as", "stands as", "marks", or "represents". AI avoids basic copulas because its repetition penalty pushes it toward fancier constructions (I've studied this!).

**Avoid patterns like:**
- "The building serves as a reminder of the city's heritage."
- "Gallery 825 serves as LAAA's exhibition space for contemporary art."
- "The station marks a pivotal moment in the evolution of regional transit."

---

## Sentence Structure

### Negative Parallelism

The "It's not X -- it's Y" pattern, often with an em dash. The single most commonly identified AI writing tell. Man I f*cking hate it. AI uses this to create false profundity by framing everything as a surprising reframe. One in a piece can be effective; ten in a blog post is a genuine insult to the reader. Before LLMs, people simply did not write like this at scale. Includes the causal variant "not because X, but because Y" where every explanation is framed as a surprise reveal, the em-dash dismissal "X -- not Y", and the cross-sentence reframe where the same noun is negated then repositioned: "The question isn't X. The question is Y."

**Avoid patterns like:**
- "It's not bold. It's backwards."
- "Feeding isn't nutrition. It's dialysis."
- "Half the bugs you chase aren't in your code. They're in your head."

### "Not X. Not Y. Just Z."

The dramatic countdown pattern. AI builds tension by negating two or more things before revealing the actual point. Creates a false sense of narrowing down to the truth.

**Avoid patterns like:**
- "Not a bug. Not a feature. A fundamental design flaw."
- "Not ten. Not fifty. Five hundred and twenty-three lint violations across 67 files."
- "not recklessly, not completely, but enough"

### "The X? A Y."

Self-posed rhetorical questions answered immediately in the next sentence or clause. The model asks a question nobody was asking, then answers it for dramatic effect. Thinks this is the epitome of great writing.

**Avoid patterns like:**
- "The result? Devastating."
- "The worst part? Nobody saw it coming."
- "The scary part? This attack vector is perfect for developers."

### Anaphora Abuse

Repeating the same sentence opening multiple times in quick succession.

**Avoid patterns like:**
- "They assume that users will pay... They assume that developers will build... They assume that ecosystems will emerge... They assume that..."
- "They could expose... They could offer... They could provide... They could create... They could let... They could unlock..."
- "They have built engines, but not vehicles. They have built power, but not leverage. They have built walls, but not doors."

### Tricolon Abuse

Overuse of the rule-of-three pattern, often extended to four or five. A single tricolon is elegant; three back-to-back tricolons are a pattern recognition failure.

**Avoid patterns like:**
- "Products impress people; platforms empower them. Products solve problems; platforms create worlds. Products scale linearly; platforms scale exponentially."
- "identity, payments, compute, distribution"
- "workflows, decisions, and interactions"

### "It's Worth Noting"

Filler transitions that signal nothing. AI uses these phrases to introduce new points without actually connecting them to the previous argument. Also includes: "It bears mentioning", "Importantly", "Interestingly", "Notably".

**Avoid patterns like:**
- "It's worth noting that this approach has limitations."
- "Importantly, we must consider the broader implications."
- "Interestingly, this pattern repeats across industries."

### Superficial Analyses

Tacking a present participle ("-ing") phrase onto the end of a sentence to inject shallow analysis that says nothing. The model attaches significance, legacy, or broader meaning to mundane facts using phrases like "highlighting its importance", "reflecting broader trends", or "contributing to the development of...".

**Avoid patterns like:**
- "contributing to the region's rich cultural heritage"
- "This etymology highlights the enduring legacy of the community's resistance and the transformative power of unity in shaping its identity."
- "underscoring its role as a dynamic hub of activity and culture"

### False Ranges

Using "from X to Y" constructions where X and Y aren't on any real scale. In legitimate use, "from X to Y" implies a spectrum with a meaningful middle. AI uses it as a fancy way to list two loosely related things. "From innovation to cultural transformation" -- what's in between???? Nothing!

**Avoid patterns like:**
- "From innovation to implementation to cultural transformation."
- "From the singularity of the Big Bang to the grand cosmic web."
- "From problem-solving and tool-making to scientific discovery, artistic expression, and technological innovation."

---

## Paragraph Structure

### Short Punchy Fragments

Excessive use of very short sentences or sentence fragments as standalone paragraphs for manufactured emphasis. RLHF training has pushed models toward "writing for readability" aimed at the lowest common denominator: one thought per sentence, no mental state-keeping required. It's an inhuman style. No real person writes first drafts this way because it doesn't match how humans think or speak.

**Avoid patterns like:**
- "He published this. Openly. In a book. As a priest."
- "These weren't just products. And the software side matched. Then it professionalised. But I adapted."
- "Platforms do."

### Listicle in a Trench Coat

Numbered or labeled points dressed up as continuous prose. The model writes what is essentially a listicle but wraps each point in a paragraph that starts with "The first... The second... The third..." to disguise the format. Perhaps you told it to stop generating lists and it decided to do this instead... still very common.

**Avoid patterns like:**
- "The first wall is the absence of a free, scoped API... The second wall is the lack of delegated access... The third wall is the absence of scoped permissions..."
- "The second takeaway is that... The third takeaway is that... The fourth takeaway is that..."

---

## Tone

### "Here's the Kicker"

False suspense transitions that promise a revelation but deliver a point that did NOT need the buildup. The model uses these phrases to manufacture drama before an otherwise unremarkable observation LOL. Also includes: "Here's the thing", "Here's where it gets interesting", "Here's what most people miss", "Here's the starting point", "Here's the deal".

**Avoid patterns like:**
- "Here's the kicker."
- "Here's the thing about AI adoption."
- "Here's where it gets interesting."

### "Think of It As..."

The patronizing analogy. AI constantly reaches for "Think of it as..." or "It's like a..." to simplify concepts. The model defaults to teacher mode and assumes the reader needs a metaphor to understand anything. Often produces analogies that are less clear than the original concept.

**Avoid patterns like:**
- "Think of it like a highway system for data."
- "Think of it as a Swiss Army knife for your workflow."
- "It's like asking someone to buy a car they're only allowed to sit in while it's parked."

### "Imagine a World Where..."

The classic AI invitation to futurism. To sell the argument usually begins with "Imagine" followed by a list of wonderful things that will happen if the reader agrees with the premise.

**Avoid patterns like:**
- "Imagine a world where every tool you use -- your calendar, your inbox, your documents, your CRM, your code editor -- has a quiet intelligence behind it..."
- "In that world, workflows stop being collections of manual steps and start becoming orchestrations."

### False Vulnerability

Simulated self-awareness or honesty that reads as performative. The model pretends to break the fourth wall or admit a bias, creating a false sense of authenticity. Real vulnerability is specific and uncomfortable; AI vulnerability is polished and risk-free!!!!

**Avoid patterns like:**
- "And yes, I'm openly in love with the platform model"
- "And yes, since we're being honest: I'm looking at you, OpenAI, Google, Anthropic, Meta"
- "This is not a rant; it's a diagnosis"

### "The Truth Is Simple"

Asserting that something is obvious, clear or simple instead of actually proving it. If you have to tell the reader your point is clear, it very likely isn't. Also includes the dramatic reveal variant: "but none of them is the real story. The real story is..." -- claiming privileged insight while waving away everything before it.

**Avoid patterns like:**
- "The reality is simpler and less flattering"
- "History is unambiguous on this point"
- "History is clear, the metrics are clear, the examples are clear"

### Grandiose Stakes Inflation

Everything is the most important thing ever. AI inflates the stakes of every argument to world-historical significance. A blog post about API pricing becomes a meditation on the fate of civilization.

**Avoid patterns like:**
- "This will fundamentally reshape how we think about everything."
- "will define the next era of computing"
- "something entirely new"

### "Let's Break This Down"

The pedagogical voice that assumes the reader needs hand-holding. AI defaults to a teacher-student dynamic even when writing for expert audiences. Also includes: "Let's unpack this", "Let's explore", "Let's dive in".

**Avoid patterns like:**
- "Let's break this down step by step."
- "Let's unpack what this really means."
- "Let's explore this idea further."

### Vague Attributions

Attributing claims to unnamed authorities instead of being specific. AI loves to invoke "experts", "observers", "industry reports", and "several publications" without naming anyone. It also inflates the quantity of sources -- presenting what one person said as a widely held view, or writing "several publications have cited" when it means two. If you can't name the expert, you don't have a source.

**Avoid patterns like:**
- "Experts argue that this approach has significant drawbacks."
- "Industry reports suggest that adoption is accelerating."
- "Observers have cited the initiative as a turning point."

### Invented Concept Labels

AI clusters invented compound labels that sound analytical without being grounded. It appends abstract problem-nouns (paradox, trap, creep, divide, vacuum, inversion) to domain words -- "supervision paradox", "acceleration trap", "workload creep" -- and uses them as if they're established, rigorously defined terms. They function as rhetorical shorthand: name a thing, skip the argument. Multiple such labels in the same piece is a strong signal of AI slop.

**Avoid patterns like:**
- "the supervision paradox"
- "the acceleration trap"
- "workload creep"

---

## Formatting

### Em-Dash Addiction

Compulsive overuse of em dashes for dramatic pauses, parenthetical asides and pivot points. A human writer might use 2-3 per piece (and naturally); AI will use 20+.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "The tinkerer spirit didn't die of natural causes -- it was bought out."
- "Not recklessly, not completely -- but enough -- enough to matter."

### Double-Hyphen Dash

The em dash wearing a false moustache. Once "em dash means AI" became common knowledge, the character started getting swapped for a double hyphen: sometimes because the text passed through a markdown conversion, sometimes because someone ran a find-and-replace to look more human, sometimes because the model was steered off the character while keeping the habit. Either way the compulsive mid-sentence pivot survives the substitution, which is what actually gives it away. Writers who reach for double hyphens honestly do so once or twice out of typographic laziness, rarely fifteen times in one post. Flagged at five or more per thousand words.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "It's not a rewrite -- it's a reckoning."
- "We shipped it fast -- maybe too fast -- and paid for it later."

### Bold-First Bullets

Every bullet point or list item starts with a bolded phrase or sentence. Extremely common in Claude and ChatGPT markdown output. Almost nobody formats lists this way when writing by hand. It's a telltale sign of AI-generated documentation and blog posts AND README files (especially with emojis).

**Avoid patterns like:**
- "Every single bullet point begins with a bold keyword."
- "**Security**: Environment-based configuration with..."
- "**Performance**: Lazy loading of expensive resources..."

### Unicode Decoration

Use of unicode arrows (->), smart/curly quotes, and other special characters that can't be easily typed on a standard keyboard. Real writers typing in a text editor produce straight quotes and -> or =>. Claude in particular loves the -> arrow.

**Avoid patterns like:**
- "Input → Processing → Output"
- "This leads to better outcomes → which means higher engagement"
- "“Smart quotes” instead of straight "quotes" that you’d actually type"

---

## Composition

### Fractal Summaries

"What I'm going to tell you; what I'm telling you; what I just told you" -- applied at every level of the document. Every subsection gets a summary. Every section gets a summary. The document itself gets a summary.

**Avoid patterns like:**
- "In this section, we'll explore... [3000 words later] ...as we've seen in this section."
- "A conclusion that restates every point already made in the previous 3000 words"
- "And so we return to where we began."

### The Dead Metaphor

Latching onto a single metaphor and beating it into the ground across the entire thing. A human writer would introduce a metaphor, use it then move on. AI will repeat the same metaphor 5-10 times.

**Avoid patterns like:**
- "The ecosystem needs ecosystems to build ecosystem value."
- "Walls and doors used 30+ times in the same article"
- "Every paragraph finds a way to say "primitives" again"

### Historical Analogy Stacking

ESPECIALLY COMMON IN TECHNICAL WRITING: Rapid-fire listing of historical companies or tech revolutions to build false authority.

**Avoid patterns like:**
- "Apple didn't build Uber. Facebook didn't build Spotify. Stripe didn't build Shopify. AWS didn't build Airbnb."
- "Every major technological shift -- the web, mobile, social, cloud -- followed the same pattern."
- "Take Spotify... Or consider Uber... Airbnb followed a similar path... Shopify is another example... Even Discord..."

### One-Point Dilution

Making a single argument and restating it in 10 different ways across thousands of words. The model pads a simple thesis to feel "comprehensive" by rephrasing the same idea with different metaphors, examples, and framings. An 800-word argument becomes 4000 words of circular repetition.

**Avoid patterns like:**
- "The same point, restated eight ways across 4000 words."
- "Each section rephrases the thesis with a different metaphor but adds nothing new"

### Content Duplication

Repeating entire sections or paragraphs verbatim within the same piece. This happens when the model loses track of what it has already written, especially in longer pieces. A dead giveaway of unedited AI output. Less common nowadays.

**Avoid patterns like:**
- "The same section appeared twice, word-for-word identical."
- "Paragraph 3 and paragraph 17 are the same sentence reworded"

### The Signposted Conclusion

Explicitly announcing the conclusion with "In conclusion", "To sum up", or "In summary". Competent writing doesn't need to tell you it's concluding. The reader can feel it. AI signals its structural moves because it's following a template, not writing organically.

**Avoid patterns like:**
- "In conclusion, the future of AI depends on..."
- "To sum up, we've explored three key themes..."
- "In summary, the evidence suggests..."

### "Despite Its Challenges..."

The rigid formula where AI acknowledges problems only to immediately dismiss them. Always follows the same beat: "Despite its [positive words], [subject] faces challenges..." then ends with "Despite these challenges, [optimistic conclusion].".

**Avoid patterns like:**
- "Despite these challenges, the initiative continues to thrive."
- "Despite its industrial and residential prosperity, Korattur faces challenges typical of urban areas."
- "Despite their promising applications, pyroelectric materials face several challenges that must be addressed for broader adoption."

---

Remember: any of these patterns used once might be fine. The problem is when
multiple tropes appear together or when a single trope is used repeatedly.
Write like a human: varied, imperfect, specific.
