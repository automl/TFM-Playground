"""Regression-specific training callback for experiment tracking."""

from sklearn.metrics import root_mean_squared_error

from tfmplayground.priors.experiments.new_evaluation import get_openml_predictions
from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.utils import get_default_device


class RegressionTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks RMSE on toy tasks and stores the final RMSE and loss history."""

    def __init__(self, tasks, model_name="Model", eval_every: int = 1):
        self.tasks = tasks
        self.model_name = model_name
        self.eval_every = max(1, int(eval_every))
        self.final_rmse = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.rmse_history = []  # may contain None for skipped epochs
        self.task_rmse_values = {}
        self.epoch_history = []
        self.epoch_times = []

    def on_epoch_end(
        self, epoch: int, epoch_time: float, loss: float, model, dist=None, **kwargs
    ):
        # Always track loss per epoch
        self.epoch_history.append(epoch)
        self.epoch_times.append(epoch_time)
        self.loss_history.append(loss)

        # Optionally skip expensive evaluation
        if (epoch % self.eval_every) != 0:
            self.rmse_history.append(None)
            print(
                f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
                f"mean loss {loss:5.2f} | eval skipped (every {self.eval_every})",
                flush=True,
            )
            return

        # Use the full NanoTabPFNRegressor which handles the distribution
        regressor = NanoTabPFNRegressor(model=model, dist=dist, device=self.device)
        per_fold_dataset_predictions = get_openml_predictions(
            model=regressor, tasks=self.tasks, classification=False
        )
        rmse_values = []
        for dataset_name, per_fold_predictions in per_fold_dataset_predictions.items():
            fold_rmse_values = []
            for fold_dict in per_fold_predictions:
                fold_rmse = root_mean_squared_error(
                    fold_dict["y_true"], fold_dict["y_pred"]
                )
                fold_rmse_values.append(fold_rmse)
            avg_fold_rmse = (
                sum(fold_rmse_values) / len(fold_rmse_values)
                if len(fold_rmse_values)
                else float("nan")
            )
            rmse_values.append(avg_fold_rmse)
            if dataset_name not in self.task_rmse_values:
                self.task_rmse_values[dataset_name] = []
            self.task_rmse_values[dataset_name].append(fold_rmse_values)
        avg_rmse = (
            sum(rmse_values) / len(rmse_values) if len(rmse_values) else float("nan")
        )
        self.final_rmse = avg_rmse
        self.rmse_history.append(avg_rmse)

        print(
            f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
            f"mean loss {loss:5.2f} | avg RMSE {avg_rmse:.3f}",
            flush=True,
        )
