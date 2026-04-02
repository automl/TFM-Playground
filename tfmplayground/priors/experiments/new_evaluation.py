import openml
import torch
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.preprocessing import LabelEncoder

from tfmplayground.interface import NanoTabPFNRegressor, NanoTabPFNClassifier

TOY_TASKS_REGRESSION = [
    362443, # diabetes
]

TOY_TASKS_CLASSIFICATION = [
    59,   # iris
    2382, # wine
    9946, # breast_cancer
]

# we hardcode the list here because even if the tasks are cached
# openml.study.get_suite("tabarena-v0.1") might fail if there are connection issues
TABARENA_TASKS = [
    363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
    363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
    363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
    363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
    363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
    363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
    363708, 363711, 363712
]

@torch.no_grad()
def get_openml_predictions(
        *,
        model: NanoTabPFNRegressor | NanoTabPFNClassifier,
        tasks: list[int] | str = "tabarena-v0.1",
        max_n_features: int = 500,
        max_n_samples: int = 10_000,
        max_folds: int | None = None,
        classification: bool | None = None,
        cache_directory: str | None = None,
):
    """
    Evaluates a model on a set of OpenML tasks and returns predictions.

    Retrieves datasets from OpenML, applies preprocessing, and evaluates the given model on each task.
    Returns true targets, predicted labels, and predicted probabilities for each dataset.

    Args:
        model (NanoTabPFNRegressor | NanoTabPFNClassifier): A scikit-learn compatible model or classifier to be evaluated.
        tasks (list[int] | str, optional): A list of OpenML task IDs or the name of a benchmark suite.
        max_n_features (int, optional): Maximum number of features allowed for a task. Tasks exceeding this limit are skipped.
        max_n_samples (int, optional): Maximum number of instances allowed for a task. Tasks exceeding this limit are skipped.
        max_folds (int | None, optional): Maximum number of folds to evaluate per task. If None, all folds are used.
        classification (bool | None, optional): Whether the model is a classifier (True) or regressor (False). If None, it is inferred from the model type.
        cache_directory (str | None, optional): Directory to save OpenML data. If None, default cache path is used.
    Returns:
        dict: A dictionary mapping dataset names to a list (one per fold) of dicts with keys:
              "y_true", "y_pred", "y_proba" (None for regression).
    """
    if classification is None:
        classification = isinstance(model, NanoTabPFNClassifier)

    if cache_directory is not None:
        set_root_cache_directory(cache_directory)

    if isinstance(tasks, str):
        benchmark_suite = openml.study.get_suite(tasks)
        task_ids = benchmark_suite.tasks
    else:
        task_ids = tasks

    dataset_predictions = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)

        if classification and task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue # skip task, only classification
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
            continue # skip task, only regression

        dataset = task.get_dataset(download_data=False)

        n_features = dataset.qualities["NumberOfFeatures"]
        n_samples = dataset.qualities["NumberOfInstances"]
        if n_features > max_n_features or n_samples > max_n_samples:
            continue  # skip task, too big

        _, folds, _ = task.get_split_dimensions()
        if max_folds is not None:
            folds = min(folds, max_folds)
        print(f"Evaluating on task {task_id} with dataset '{dataset.name}' ({n_samples} samples, {n_features} features, {folds} folds)...")
        repeat = 0  # code only supports one repeat

        per_fold_records: list[dict] = []

        for fold in range(folds):
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )
            train_indices, test_indices = task.get_train_test_split_indices(
                fold=fold, repeat=repeat
            )
            X_train = X.iloc[train_indices].to_numpy()
            y_train = y.iloc[train_indices].to_numpy()
            X_test = X.iloc[test_indices].to_numpy()
            y_test = y.iloc[test_indices].to_numpy()

            if classification:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_proba = None
            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # binary classification
                    y_proba = y_proba[:, 1]

            rec = {
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            per_fold_records.append(rec)

        dataset_predictions[str(dataset.name)] = per_fold_records
    return dataset_predictions