"""Classification-specific training callback for experiment tracking."""

from sklearn.metrics import roc_auc_score

from tfmplayground.priors.experiments.new_evaluation import get_openml_predictions
from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.utils import get_default_device


class ClassificationTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks ROC-AUC on tasks and stores the final ROC-AUC and loss history."""

    def __init__(self, tasks, model_name="Model", eval_every: int = 1):
        self.tasks = tasks
        self.model_name = model_name
        self.eval_every = max(1, int(eval_every))
        self.final_roc_auc = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.roc_auc_history = []  # may contain None for skipped epochs
        self.task_roc_auc_values = {}  # dataset -> list[ list[fold_auc] ] (per epoch)
        self.epoch_history = []
        self.epoch_times = []

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        # Always track loss per epoch
        self.epoch_history.append(epoch)
        self.epoch_times.append(epoch_time)
        self.loss_history.append(loss)

        # Optionally skip expensive evaluation
        if (epoch % self.eval_every) != 0:
            self.roc_auc_history.append(None)
            print(
                f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
                f"mean loss {loss:5.2f} | eval skipped (every {self.eval_every})",
                flush=True,
            )
            return

        classifier = NanoTabPFNClassifier(model, self.device)
        per_fold_dataset_predictions = get_openml_predictions(
            model=classifier,
            tasks=self.tasks,
            classification=True,
        )

        dataset_auc_means = []

        for dataset_name, per_fold_predictions in per_fold_dataset_predictions.items():
            fold_auc_values = []

            for fold_dict in per_fold_predictions:
                y_true = fold_dict["y_true"]
                y_proba = fold_dict.get("y_proba", None)

                # If probabilities are missing, can't compute ROC-AUC
                if y_proba is None:
                    continue

                try:
                    # roc_auc_score supports:
                    # - binary: y_proba shape (n,) or (n,2) but we typically store (n,) positive class
                    # - multiclass: y_proba shape (n, C)
                    fold_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
                except ValueError:
                    # e.g. only one class present in y_true for this fold
                    continue

                fold_auc_values.append(fold_auc)

            avg_fold_auc = (
                sum(fold_auc_values) / len(fold_auc_values)
                if len(fold_auc_values)
                else float("nan")
            )

            dataset_auc_means.append(avg_fold_auc)

            if dataset_name not in self.task_roc_auc_values:
                self.task_roc_auc_values[dataset_name] = []
            # Store per-epoch fold values (like regression tracker does)
            self.task_roc_auc_values[dataset_name].append(fold_auc_values)

        avg_auc = (
            sum(dataset_auc_means) / len(dataset_auc_means)
            if len(dataset_auc_means)
            else float("nan")
        )

        self.final_roc_auc = avg_auc
        self.roc_auc_history.append(avg_auc)

        print(
            f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
            f"mean loss {loss:5.2f} | avg ROC-AUC {avg_auc:.3f}",
            flush=True,
        )
