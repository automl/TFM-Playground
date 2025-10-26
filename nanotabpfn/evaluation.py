import argparse

import numpy as np
import openml
import torch
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.preprocessing import LabelEncoder

from nanotabpfn.ensembles import EnsembleClassifer
from nanotabpfn.interface import NanoTabPFNRegressor, NanoTabPFNClassifier

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
        model: NanoTabPFNRegressor | NanoTabPFNClassifier | EnsembleClassifer,
        tasks: list[int] | str = "tabarena-v0.1",
        max_n_features: int = 500,
        max_n_samples: int = 10_000,
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
        classification (bool | None, optional): Whether the model is a classifier (True) or regressor (False). If None, it is inferred from the model type.
        cache_directory (str | None, optional): Directory to save OpenML data. If None, default cache path is used.
    Returns:
        dict: A dictionary where keys are dataset names and values are tuples of (true targets, predicted labels, predicted probabilities).
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
        tabarena_light = True
        if tabarena_light:
            folds = 1 # code supports multiple folds but tabarena_light only has one
        repeat = 0 # code only supports one repeat
        targets = []
        predictions = []
        probabilities = []
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
            targets.append(y_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # binary classification
                    y_proba = y_proba[:, 1]
                probabilities.append(y_proba)

        y_pred = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0) if len(probabilities) > 0 else None
        dataset_predictions[str(dataset.name)] = (targets, y_pred, probabilities)
    return dataset_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_type", type=str, choices=["regression", "classification"], required=True,
                        help="Whether to use the regressor or classifier model")
    parser.add_argument("-ensemble_size", type=int, default=None,
                        help="Set the number of preprocessors to ensemble with. If None, then no ensembling is applied.")
    parser.add_argument("-checkpoint", type=str, default=None,
                        help="Path to load the model weights from. If None, default weights are used.")
    parser.add_argument("-dist_path", type=str, default=None,
                        help="Path to load the bucket edges for the support bar distribution from. Only needed for regression.")
    parser.add_argument("-tasks", type=str, default="tabarena-v0.1",
                        choices=["tabarena-v0.1", "toy_tasks"], help="Which OpenML tasks to evaluate on.")
    parser.add_argument("-cache_directory", type=str, default=None,
                        help="Directory to save OpenML data. If None, default cache path is used.")
    parser.add_argument("-max_n_features", type=int, default=500,
                        help="Maximum number of features allowed for a task. Tasks exceeding this limit are skipped.")
    parser.add_argument("-max_n_samples", type=int, default=10_000,
                        help="Maximum number of instances allowed for a task. Tasks exceeding this limit are skipped.")
    parser.add_argument("-num_mem_chunks", type=int, default=8,
                        help="Chunks attention computation into `num_mem_chunks` many chunks to reduce memory usage.")
    args = parser.parse_args()

    if args.model_type == "classification":
        if args.ensemble_size is None:
            model = NanoTabPFNClassifier(model=args.checkpoint, num_mem_chunks=args.num_mem_chunks)
        else:
            model = EnsembleClassifer(NanoTabPFNClassifier(model=args.checkpoint, num_mem_chunks=args.num_mem_chunks, preprocess_features=False), num_preprocessors=args.ensemble_size)
    else:
        model = NanoTabPFNRegressor(model=args.checkpoint, dist=args.dist_path, num_mem_chunks=args.num_mem_chunks)
    model.model.eval()

    if args.tasks == "toy_tasks" and args.model_type == "regression":
        tasks = TOY_TASKS_REGRESSION
    elif args.tasks == "toy_tasks" and args.model_type == "classification":
        tasks = TOY_TASKS_CLASSIFICATION
    else:
        tasks = args.tasks

    predictions = get_openml_predictions(
        model=model, tasks=tasks, max_n_features=args.max_n_features, max_n_samples=args.max_n_samples,
        classification=(args.model_type=="classification"), cache_directory=args.cache_directory
    )

    average_score = 0
    for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
        if args.model_type == "classification":
            acc = balanced_accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            average_score  += auc
            print(f"Dataset: {dataset_name} | ROC AUC: {auc:.4f} | Balanced Accuracy: {acc:.4f}")
        else:
            r2 = r2_score(y_true, y_pred)
            average_score  += r2
            print(f"Dataset: {dataset_name} | R2: {r2:.4f}")
    average_score /= len(predictions)
    print(f"Average {'ROC AUC' if args.model_type == 'classification' else 'R2'}: {average_score:.4f}")
