from typing import TYPE_CHECKING, Tuple, Dict, List
from matplotlib import pyplot as plt
import os
import functools
from sklearn.linear_model import LinearRegression, LogisticRegression
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib as mpl
import openml
from tqdm import tqdm
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
"""
=================== DATA LOADING AND PREPROCESSING ===================
"""

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())), # in case numeric columns are stored as strings
        ("imputer", SimpleImputer(strategy="mean")),
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor

def get_openml_datasets(
        max_features_eval: int | None,
        new_instances_eval: int | None,
        target_classes_filter: int | None,
        eval_subsample_features: int | None,
        eval_subsample_samples: int | None,
        seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML tabarena datasets with optional feature and row subsampling.

    Parameters
    ----------
    max_features_eval : int | None
        Maximum number of features a dataset may have to be included. None = no filter.
    new_instances_eval : int | None
        Maximum number of instances to keep via stratified subsampling. None = no subsampling.
    target_classes_filter : int | None
        Maximum number of target classes (0 = regression). None = no filter.
    eval_subsample_features : int | None
        If set and the dataset has more features than this value, randomly
        subsample down to this many features (seeded).
    eval_subsample_samples : int | None
        If set and the dataset has more rows than this value, stratified-
        subsample down to this many rows (seeded).
    seed : int
        Global random seed used for all stochastic operations.

    Returns
    -------
    dict mapping dataset name -> (X, y) as numpy arrays.
    """
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
        363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
        363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
        363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
        363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
        363708, 363711, 363712,
    ]  # TabArena v0.1

    classification: bool = target_classes_filter is None or target_classes_filter > 0

    datasets = {}
    for task_counter, task_id in enumerate(task_ids):
        task = openml.tasks.get_task(task_id, download_splits=False)

        # ── task-type filter ────────────────────────────────────────────────
        if classification and task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
            continue

        dataset = task.get_dataset(download_data=False)

        # ── quality filter ──────────────────────────────────────────────────
        q = dataset.qualities
        if max_features_eval is not None and q["NumberOfFeatures"] > max_features_eval:
            continue
        if new_instances_eval is not None and q["NumberOfInstances"] > new_instances_eval:
            continue
        if target_classes_filter is not None and q["NumberOfClasses"] > target_classes_filter:
            continue
        if dataset.qualities["MinorityClassPercentage"] <= 2.5:
            continue

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        # ── feature subsampling ─────────────────────────────────────────────
        len_features = X.shape[1]
        if eval_subsample_features is not None and len_features > eval_subsample_features:
            rng = np.random.default_rng(seed)
            feature_choices = rng.choice(len_features, size=eval_subsample_features, replace=False)
            X = X.iloc[:, feature_choices]

        # ── row subsampling ─────────────────────────────────────────────────
        if eval_subsample_samples is not None and eval_subsample_samples < len(y):
            y_stratify_sub = y if classification else pd.qcut(y, q=5, labels=False, duplicates="drop")
            _, X, _, y = train_test_split(
                X, y,
                test_size=eval_subsample_samples,
                stratify=y_stratify_sub,
                random_state=seed,
            )
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        # ── preprocessing & encoding ────────────────────────────────────────
        X = X.to_numpy(copy=True)
        y = y.to_numpy(copy=True)

        if classification:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        else:
            target_scaler = StandardScaler()
            y = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        preprocessor = get_feature_preprocessor(X)
        X = preprocessor.fit_transform(X)

        datasets[dataset.name] = (X, y)

    task_counter += 1
    return datasets


"""
=================== EVALUATION ===================
"""


def eval_model(model, datasets, classification: bool):
    """Evaluates a model on multiple datasets and returns metrics"""
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    metrics = {}
    avg_metrics = {}
    for dataset_name, (X, y) in tqdm(datasets.items(), desc=f"Evaluating {model}", total=len(datasets), leave=False):
        targets = []
        probabilities = []
        
        y_stratify = y if classification else pd.qcut(y, q=5, labels=False, duplicates='drop')
        for train_idx, test_idx in _skf.split(X, y_stratify):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test  = y[train_idx], y[test_idx]
            targets.append(y_test)
            model.fit(X_train, y_train)

            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # binary classification with neural network
                    y_proba = y_proba[:, 1]
                probabilities.append(y_proba)
            else:
                y_pred = model.predict(X_test)
                probabilities.append(y_pred)
    
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)

        if classification:
            metrics[f"{dataset_name}/roc_auc"] = roc_auc_score(targets, probabilities, multi_class="ovr")
        else:
            metrics[f"{dataset_name}/rmse"] = root_mean_squared_error(targets, probabilities)
    
    metric_names = list({key.split("/")[-1] for key in metrics.keys()})
    for metric_name in metric_names:
        avg_metric = np.mean([metrics[key] for key in metrics.keys() if key.endswith(metric_name)])
        avg_metrics[f"{metric_name}"] = float(avg_metric)
    
    return metrics, avg_metrics

"""
=================== PLOTTING ===================
"""

def plot_runs(
        ax: plt.Axes, 
        runs: list[pd.DataFrame], 
        metric: str, 
        baselines: pd.DataFrame = None,
        baselines_std: pd.DataFrame = None,
        show_legend: bool = True,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        show_xtics: bool = True
        ):
    """
    Plots the run for a given metric and adds baselines

    `runs` is a list of dataframes where each dataframe corresponds 
    to a run from a model with the same config but a different seed.
    Each dataframe needs to have a `"training_time"` column and a `metric`column.

    `baselines` is a DataFrame with metric columns and rows whose index correspond to
    a ML algorithm.
    """
    colors = sns.color_palette("tab10")[1:]
    linestyles = [
        '--',      # dashed
        '-.',      # dash-dot
        ':',       # dotted
        (0, (3, 1, 1, 1)),  # dash-dot-dot
        (0, (5, 5))         # spaced dash
    ]

    training_times = [run["training_time"].tolist() for run in runs]
    training_times = sorted(set([item for sublist in training_times for item in sublist]))
    shared_time_runs = []
    for run in runs:
        run = run.copy()
        run = run[[metric, "training_time"]].set_index("training_time").reindex(training_times)
        run = run.interpolate()
        shared_time_runs.append(run)
    all_runs = pd.concat(shared_time_runs, axis=1).dropna()
    
    # plot mean and std of all runs or single run if only one run
    mean = all_runs.mean(axis=1)
    ax.plot(mean.index, mean, label="nanoTabPFN", zorder=2, color="blue")
    if all_runs.shape[1] > 1: # more than one run
        std = all_runs.std(axis=1)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, zorder=2)

    # plot horizontal lines for baselines
    if baselines is not None:
        for i ,(baseline_name, baseline_value) in enumerate(baselines[metric].items()):
            # draw a horizontal line that ends at the same x as the runs
            color = colors[i % len(colors)]
            ax.plot([0, max(training_times)], [baseline_value, baseline_value], label=baseline_name, alpha=0.7, linestyle=linestyles[i], color=color, zorder=1)
            if baselines_std is not None and baseline_name in baselines_std.index:
                    std = baselines_std.loc[baseline_name, metric]
                    ax.fill_between([0, max(training_times)], [baseline_value - std, baseline_value - std], [baseline_value + std, baseline_value + std], alpha=0.2, zorder=1)
    
    # Plot Style
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="y", length=0)
    if not show_xtics:
        ax.tick_params(axis="x", length=0)
    if show_xlabel:
        ax.set_xlabel("Training time (seconds)")
    if show_ylabel:
        ax.set_ylabel(metric.split("/")[-1])
    max_time = max(training_times)
    ax.set_xlim(0, max_time)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1)
    
    # order legend entries by their y-value at the end of the plot
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        label_y_values = {}
        for handle, label in zip(handles, labels):
            if isinstance(handle, plt.Line2D):
                y_data = handle.get_ydata()
                label_y_values[label] = y_data[-1]
        sorted_labels = sorted(label_y_values.items(), key=lambda x: x[1], reverse=True)
        sorted_handles = [handle for label, _ in sorted_labels for handle, lbl in zip(handles, labels) if lbl == label]
        sorted_labels = [label for label, _ in sorted_labels]
        ax.legend(sorted_handles, sorted_labels)
        
    # remove border
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_run_grid(runs: list[pd.DataFrame], baselines: pd.DataFrame = None, baselines_std: pd.DataFrame = None, metric: str = "roc_auc") -> Tuple[plt.Figure, np.ndarray]:
    """Plots the runs in a grid of metrics x datasets"""
    # drop all columns without "/" for dataset/metric format
    datasets = list(set([col.split("/")[0] for col in runs[0].columns if "/" in col]))
    figsize = (len(datasets) * 4, 4.6)
    fig, axs = plt.subplots(1, len(datasets), figsize=figsize, sharex=True, sharey=True, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.1)
    # Plot each metric and dataset
    for j, dataset in enumerate(datasets):
        ax = axs[j]
        plot_runs(ax, runs, f"{dataset}/{metric}", baselines, baselines_std, show_legend=False, show_xlabel=False, show_ylabel=(j==0))
        ax.set_title(dataset)
    fig.supxlabel("Training Time (seconds)")
    
    # y-axis and x-axis labels should have the same size as supxlabel
    for ax in axs.flatten():
        font_size = fig.texts[-1].get_fontsize() 
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
    
    # Create a single legend for the entire figure
    legend_handels_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axs.flatten()]
    legend_handels_labels = functools.reduce(lambda a, b: a + b, legend_handels_labels)
    unique = dict([(label, handle) for (handle, label) in legend_handels_labels])
    fig.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=3)   
    return fig, axs


def plot_dataset_runs_large(
    runs: list[pd.DataFrame],
    baselines: pd.DataFrame = None,
    baselines_std: pd.DataFrame = None,
    metric: str = "roc_auc",
    figsize: tuple = (6, 4.5),):

    """
    This is the large-figure version of plot_run_grid(), creating one figure per dataset.
    """

    datasets = list(set([col.split("/")[0] for col in runs[0].columns if "/" in col]))
    
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        # Plot the training run and baselines for this dataset
        dataset_metric = f"{dataset}/{metric}"

        plot_runs(
            ax=ax,
            runs=runs,
            metric=dataset_metric,
            baselines=baselines,
            baselines_std=baselines_std,
            show_legend=True,
            show_xlabel=True,
            show_ylabel=True,
        )

        ax.set_title(dataset)

        # Show and close the figure
        plt.show()
        plt.close(fig)


def get_baseline_results(
    open_ml_datasets_kwargs: dict,
    num_seeds: int = 1,
    include_tabpfn: bool = False,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], pd.DataFrame, pd.DataFrame]:
    """
    Reproducing the training and evalaution of the NanoTabPFNPlayground paper notebook.
    """
    NUM_SEEDS = num_seeds
    # NUM_SEEDS = 20 # If you want to reproduce the paper results, use 20 seeds
    datasets = get_openml_datasets(**open_ml_datasets_kwargs)

    classification = open_ml_datasets_kwargs['target_classes_filter'] > 0

    if include_tabpfn:
        raise NotImplementedError
        # from tabpfn import TabPFNClassifier
        # from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig


        # no_preprocessing_inference_config = ModelInterfaceConfig(
        #     FINGERPRINT_FEATURE=False,
        #     PREPROCESS_TRANSFORMS=[PreprocessorConfig(name='none')]
        # )

    if classification:
        baseline_models = {
            # "TabPFN v2": [TabPFNClassifier(random_state=i) for i in range(NUM_SEEDS)],
            # "TabPFN v2 (no preprocessing)": [TabPFNClassifier(inference_config=no_preprocessing_inference_config, n_estimators=1, random_state=i) for i in range(NUM_SEEDS)],
            "Random Forest": [RandomForestClassifier(random_state=i) for i in range(NUM_SEEDS)],
            "K-Nearest Neighbors": [KNeighborsClassifier()],
            "Decision Tree": [DecisionTreeClassifier(random_state=i) for i in range(NUM_SEEDS)],
            "Linear" : [LogisticRegression(max_iter=1000, ) for i in range(NUM_SEEDS)],
        }
    else:
        baseline_models = {
            # "TabPFN v2": [TabPFNRegressor(random_state=i) for i in range(NUM_SEEDS)],
            # "Random Forest": [RandomForestRegressor(random_state=i) for i in range(NUM_SEEDS)],
            # "K-Nearest Neighbors": [KNeighborsRegressor()],
            "Decision Tree": [DecisionTreeRegressor(random_state=i) for i in range(NUM_SEEDS)],
            # "Linear" : [LinearRegression()],
        }

    baseline_models_eval = {name: [eval_model(model, datasets=datasets, classification=classification)[0] for model in models] for name, models in baseline_models.items()}

    def apply_aggregation(eval_results: dict, func=np.mean):
        aggregated_result = {}
        for result in eval_results:
            for metric, value in result.items():
                if metric not in aggregated_result:
                    aggregated_result[metric] = []
                aggregated_result[metric].append(value)
        for metric in aggregated_result:
            aggregated_result[metric] = func(aggregated_result[metric])
        return aggregated_result

    baselines = pd.DataFrame({
        name: apply_aggregation(models, np.mean) for name, models in baseline_models_eval.items()
    }).T

    baselines_std = pd.DataFrame({
        name: apply_aggregation(models, np.std) for name, models in baseline_models_eval.items()
    }).T

    return datasets, baselines, baselines_std


def extract_final_variant_score(
    histories,
    metric: str = "roc_auc",
    epoch: int = 20,
) -> float:
    
    """
    Extract the logged metric from the latest run at the selected epoch.
    """

    # Use the latest run in the histories list
    if isinstance(histories, list):
        run = histories[-1]
    else:
        run = histories

    available_columns = list(run.columns)

    if metric not in available_columns:
        raise ValueError(f"Column '{metric}' not found. Available columns: {available_columns}")

    score = run.loc[epoch, metric]
    score = float(score)

    return score


def extract_classical_baseline_scores(baselines, metric: str = "roc_auc") -> pd.DataFrame:

    """
    Reshape the already-computed classical baseline results from get_baseline_results() into the format needed for the final ROC-AUC plot.
    """

    baseline_df = baselines.copy()

    # Case 1: the metric is already available as one average column
    if metric in baseline_df.columns:
        scores = baseline_df[metric]

    # Case 2: the dataframe has one metric column per dataset, such as: dataset_1/roc_auc, dataset_2/roc_auc, ...
    else:
        metric_cols = []

        for col in baseline_df.columns:
            if isinstance(col, str) and col.endswith(f"/{metric}"):
                metric_cols.append(col)

        if len(metric_cols) == 0:
            raise ValueError(f"Could not find metric '{metric}' in baselines. Available columns are: {list(baseline_df.columns)}")

        # Average the classical baseline performance across datasets
        scores = baseline_df[metric_cols].mean(axis=1)

    classical_scores = pd.DataFrame({"model": scores.index.astype(str),
                                     "final_roc_auc": scores.values,
                                     "group": "Classical baseline"})

    return classical_scores


def update_and_build_result_scores_for_plot(
    variant_name: str,
    histories,
    baselines,
    metric: str = "roc_auc",
    results_csv: str = "../logs/store/final_results/final_roc_auc_all_models_subsample.csv",
    # results_csv: str = "../logs/store/final_results/final_roc_auc_all_models_full.csv"
    run_scores_csv: str = "../logs/store/final_results/variant_run_scores_subsample.csv",
    # run_scores_csv: str = "../logs/store/final_results/variant_run_scores_full.csv"
    epoch: int = 20,
):
    
    """
    Save the latest run result for one NanoTabPFN variant

    Each time this function is called, it adds one row to the run-level table: model, run_id, epoch, final_roc_auc, group 
    Then it summarizes all saved runs

    Finally, it combines NanoTabPFN variant summaries with classical baselines
    """

    # Create output folders if they do not exist
    results_folder = os.path.dirname(results_csv)
    os.makedirs(results_folder, exist_ok=True)

    run_scores_folder = os.path.dirname(run_scores_csv)
    os.makedirs(run_scores_folder, exist_ok=True)

    # Extract score from the latest run
    final_score = extract_final_variant_score(histories=histories, metric=metric, epoch=epoch)

    # Load previous run-level scores if they exist
    if os.path.exists(run_scores_csv):
        run_scores = pd.read_csv(run_scores_csv)
    else:
        run_scores = pd.DataFrame(columns=["model", "run_id", "epoch", "final_roc_auc", "group"])

    # Find next run_id for this variant
    previous_rows = run_scores[run_scores["model"] == variant_name]

    if len(previous_rows) == 0:
        run_id = 1
    else:
        run_id = int(previous_rows["run_id"].max()) + 1
        if run_id == 6:
            raise ValueError(f"The maximum number of runs per variant is 5")

    # Create one new row for this run
    new_row = pd.DataFrame({"model": [variant_name],
                            "run_id": [run_id],
                            "epoch": [epoch],
                            "final_roc_auc": [final_score],
                            "group": ["NanoTabPFN variant"]})

    # Append new run
    run_scores = pd.concat([run_scores, new_row], ignore_index=True)

    # Save run-level table
    run_scores.to_csv(run_scores_csv, index=False)

    # Summarize NanoTabPFN variants across runs
    variant_summary = (run_scores.groupby("model")["final_roc_auc"].agg(["mean", "std", "count"]).reset_index())

    variant_summary = variant_summary.rename(columns={"mean": "final_roc_auc",
                                                      "std": "std",
                                                      "count": "n_runs"})

    variant_summary["se"] = (variant_summary["std"] / np.sqrt(variant_summary["n_runs"]))

    variant_summary["group"] = "NanoTabPFN variant"

    variant_summary = variant_summary[["model", "final_roc_auc", "std", "se", "n_runs", "group"]]

    # Extract classical baseline scores
    classical_scores = extract_classical_baseline_scores(baselines=baselines, metric=metric)

    classical_scores["std"] = np.nan
    classical_scores["se"] = np.nan
    classical_scores["n_runs"] = np.nan

    classical_scores = classical_scores[["model", "final_roc_auc", "std", "se", "n_runs", "group"]]

    # Combine variants and classical baselines
    result_scores = pd.concat([variant_summary, classical_scores], ignore_index=True)

    result_scores = sort_results_by_model_order(result_scores)

    result_scores.to_csv(results_csv, index=False)

    print(
        f"Saved {variant_name}, "
        f"run {run_id}: "
        f"final_roc_auc={final_score:.4f}"
    )

    return result_scores


def extract_per_dataset_variant_scores(histories, metric: str = "roc_auc", epoch: int = 20) -> pd.Series:
    
    """Extract per-dataset scores from the latest run at the selected epoch"""

    if isinstance(histories, list):
        run = histories[-1]
    else:
        run = histories

    metric_cols = []

    for col in run.columns:
        if isinstance(col, str) and col.endswith(f"/{metric}"):
            metric_cols.append(col)

    if len(metric_cols) == 0:
        raise ValueError(f"No per-dataset metric columns ending with '/{metric}' found.")

    scores = run.loc[epoch, metric_cols].copy()

    dataset_names = []

    for col in scores.index:
        dataset_name = col.replace(f"/{metric}", "")
        dataset_names.append(dataset_name)

    scores.index = dataset_names

    return scores


def extract_per_dataset_classical_scores(baselines, metric: str = "roc_auc"):

    """Convert classical baselines into per-dataset wide format"""

    metric_cols = []

    for col in baselines.columns:
        if isinstance(col, str) and col.endswith(f"/{metric}"):
            metric_cols.append(col)

    if len(metric_cols) == 0:
        raise ValueError(f"No columns ending with '/{metric}' found in baselines. Available columns: {list(baselines.columns)}")

    classical_per_dataset = baselines[metric_cols].copy()

    new_column_names = []

    for col in classical_per_dataset.columns:
        dataset_name = col.replace(f"/{metric}", "")
        new_column_names.append(dataset_name)

    classical_per_dataset.columns = new_column_names

    classical_per_dataset = classical_per_dataset.T
    classical_per_dataset.index.name = "dataset"

    return classical_per_dataset


def update_per_dataset_variant_csv(
    variant_name: str,
    histories,
    metric: str = "roc_auc",
    epoch: int = 20,
    results_csv: str = "../logs/store/final_results/per_dataset_variant_roc_auc_subsample.csv",
    # results_csv: str = "../logs/store/final_results/per_dataset_variant_roc_auc_full.csv"
    run_scores_csv: str = "../logs/store/final_results/per_dataset_variant_run_scores_subsample.csv",
    # run_scores_csv: str = "../logs/store/final_results/per_dataset_variant_run_scores_full.csv"
):
    """
    Save per-dataset ROC-AUC scores for the latest run of one variant
    Each call adds multiple rows to the run-level table: model, run_id, epoch, dataset, score
    Then it summarizes all saved runs
    """

    results_folder = os.path.dirname(results_csv)
    os.makedirs(results_folder, exist_ok=True)

    run_scores_folder = os.path.dirname(run_scores_csv)
    os.makedirs(run_scores_folder, exist_ok=True)

    # Extract per-dataset scores from latest run
    variant_scores = extract_per_dataset_variant_scores(histories=histories, metric=metric, epoch=epoch)

    # Load previous per-dataset run-level scores
    if os.path.exists(run_scores_csv):
        run_scores = pd.read_csv(run_scores_csv)
    else:
        run_scores = pd.DataFrame(
            columns=["model", "run_id", "epoch", "dataset", "score"])

    # Find next run_id for this variant
    previous_rows = run_scores[run_scores["model"] == variant_name]

    if len(previous_rows) == 0:
        run_id = 1
    else:
        run_id = int(previous_rows["run_id"].max()) + 1

    # Create one row per dataset for this run
    rows = []

    for dataset_name, score in variant_scores.items():
        rows.append({
            "model": variant_name,
            "run_id": run_id,
            "epoch": epoch,
            "dataset": dataset_name,
            "score": float(score),
        })

    new_rows = pd.DataFrame(rows)

    # Append new per-dataset run scores
    run_scores = pd.concat([run_scores, new_rows], ignore_index=True)

    # Save run-level per-dataset table
    run_scores.to_csv(run_scores_csv, index=False)

    # Summarize mean per dataset across runs
    mean_scores = (run_scores.groupby(["dataset", "model"])["score"].mean().reset_index())

    result_df = mean_scores.pivot(index="dataset", columns="model", values="score")

    # Reorder columns using the existing model-order helper
    column_order_df = pd.DataFrame({"model": result_df.columns})

    column_order_df = sort_results_by_model_order(column_order_df)

    ordered_columns = column_order_df["model"].tolist()

    result_df = result_df[ordered_columns]
    result_df.index.name = "dataset"

    result_df.to_csv(results_csv)

    print(
        f"Saved per-dataset scores for {variant_name}, "
        f"run {run_id}"
    )

    return result_df


def combine_per_dataset_variant_and_classical_scores(variant_per_dataset_scores: pd.DataFrame, classical_per_dataset_scores: pd.DataFrame) -> pd.DataFrame:
    
    """Combine per-dataset scores from NanoTabPFN variants and classical baselines"""

    variant_scores = variant_per_dataset_scores.copy()
    classical_scores = classical_per_dataset_scores.copy()

    # Use dataset as index if it is stored as a column
    if "dataset" in variant_scores.columns:
        variant_scores = variant_scores.set_index("dataset")

    if "dataset" in classical_scores.columns:
        classical_scores = classical_scores.set_index("dataset")

    # Remove unnamed columns if they exist
    for col in variant_scores.columns:
        if "Unnamed" in str(col):
            variant_scores = variant_scores.drop(columns=[col])

    for col in classical_scores.columns:
        if "Unnamed" in str(col):
            classical_scores = classical_scores.drop(columns=[col])

    # Keep only datasets that appear in both tables
    common_datasets = variant_scores.index.intersection(classical_scores.index)

    variant_scores = variant_scores.loc[common_datasets]
    classical_scores = classical_scores.loc[common_datasets]

    # Combine columns
    combined_per_dataset_scores = pd.concat([variant_scores, classical_scores], axis=1)

    # Reorder columns using the existing model-order helper
    column_order_df = pd.DataFrame({"model": combined_per_dataset_scores.columns})

    column_order_df = sort_results_by_model_order(column_order_df)

    ordered_columns = column_order_df["model"].tolist()

    combined_per_dataset_scores = combined_per_dataset_scores[ordered_columns]
    combined_per_dataset_scores.index.name = "dataset"

    return combined_per_dataset_scores


def plot_final_roc_auc_all_models_vertical(
    result_scores,
    per_dataset_scores=None,
    save_path: str | None = None,
    figsize=(12, 6),
    ymin: float = 0.55,
    use_se: bool = True,
    label_offset: tuple = (0, 4),
):
    
    """Plot final ROC-AUC as a bar chart with SE or STD error bars"""

    plot_df = result_scores.copy()

    # Select error column
    if use_se:
        if "se" in plot_df.columns:
            plot_df["error"] = plot_df["se"]
        else:
            plot_df["error"] = np.nan
    else:
        if "std" in plot_df.columns:
            plot_df["error"] = plot_df["std"]
        else:
            plot_df["error"] = np.nan

    plot_df["final_roc_auc"] = pd.to_numeric(plot_df["final_roc_auc"], errors="coerce")
    plot_df["error"] = pd.to_numeric(plot_df["error"], errors="coerce")
    plot_df = plot_df.dropna(subset=["final_roc_auc"])

    # Sort model order
    plot_df = sort_results_by_model_order(plot_df)
    plot_df = plot_df.reset_index(drop=True)

    # Bar colors by group
    plot_colors = []

    for _, row in plot_df.iterrows():
        if row["group"] == "NanoTabPFN variant":
            plot_colors.append("#4C78A8")
        else:
            plot_colors.append("#F58518")

    plot_df["plot_color"] = plot_colors

    # Plot bar chart
    fig, ax = plt.subplots(figsize=figsize, dpi=180)

    x = np.arange(len(plot_df))
    bar_width = 0.78

    bars = ax.bar(x, plot_df["final_roc_auc"], width=bar_width, color=plot_df["plot_color"], edgecolor="black", linewidth=0.4, alpha=0.88, zorder=2)

    # Blurred SE / STD band + mean line
    for i in range(len(plot_df)):
        score = plot_df.loc[i, "final_roc_auc"]
        error = plot_df.loc[i, "error"]
        color = plot_df.loc[i, "plot_color"]

        if error > 0:

            # soft uncertainty band
            ax.fill_between([x[i] - bar_width / 2, x[i] + bar_width / 2], score - error, score + error, color=color, alpha=0.28, zorder=4)
    
        # mean line
        ax.hlines(y=score, xmin=x[i] - bar_width / 2, xmax=x[i] + bar_width / 2, color="black", linewidth=1.0, zorder=5)

        # value label: centered on bar, slightly above mean line
        ax.annotate(f"{score:.4f}", xy=(x[i], score), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="black", zorder=6)

    # Axis settings
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=35, ha="right", fontsize=8)

    score_max = plot_df["final_roc_auc"].max()
    score_min = plot_df["final_roc_auc"].min()

    upper_error = (plot_df["final_roc_auc"] + plot_df["error"]).max()
    lower_error = (plot_df["final_roc_auc"] - plot_df["error"]).min()

    ymax = max(score_max, upper_error) + 0.02
    ymin_final = min(ymin, score_min, lower_error - 0.015)

    ax.set_ylim(ymin_final, ymax)

    ax.set_ylabel("Final ROC-AUC", fontsize=9)
    ax.set_xlabel("Model / noise-generation configuration", fontsize=9)

    if use_se:
        title_error = "SE"
    else:
        title_error = "STD"

    ax.set_title(f"Final ROC-AUC comparison with {title_error} error bars", fontsize=11, pad=14)

    ax.grid(axis="y", linewidth=0.2, alpha=0.25, zorder=0)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=8)

    # Legend
    legend_handles = [
        mpl.patches.Patch(facecolor="#4C78A8", edgecolor="black", label="NanoTabPFN variant"),
        mpl.patches.Patch(facecolor="#F58518", edgecolor="black", label="Classical baseline")]

    ax.legend(handles=legend_handles, fontsize=8, frameon=True, loc="upper right")

    plt.tight_layout()

    # Save
    if save_path is not None:
        save_folder = os.path.dirname(save_path)

        if save_folder != "":
            os.makedirs(save_folder, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, plot_df


def plot_per_dataset_rank_heatmap(
    score_df,
    save_path: str | None = "../logs/store/final_results/per_dataset_rank_heatmap_subsample.png",
    # save_path: str | None = "../logs/store/final_results/per_dataset_rank_heatmap_full.png"
    figsize=(10, 8),
    title: str = "Per-dataset rank heatmap"
):
    
    """
    Plot a per-dataset rank heatmap.
    """

    plot_df = score_df.copy()

    rank_df = plot_df.rank(axis=1, ascending=False, method="min")
    rank_df = rank_df.astype(int)

    fig, ax = plt.subplots(figsize=figsize, dpi=180)

    im = ax.imshow(rank_df.values, aspect="auto", vmin=1, vmax=len(rank_df.columns))

    ax.set_xticks(np.arange(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns, rotation=90, ha="right", fontsize=8)

    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index, fontsize=7)

    ax.set_title(title, fontsize=11, pad=8)

    # ax.set_xlabel("Variant", fontsize=9)
    # ax.set_ylabel("Dataset", fontsize=9)

    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            rank_value = rank_df.iloc[i, j]

            ax.text(j, i, str(rank_value), ha="center", va="center", fontsize=5.5)

    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(f"Rank within dataset: 1 = best, {len(rank_df.columns)} = worst", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    if save_path is not None:
        save_folder = os.path.dirname(save_path)
        os.makedirs(save_folder, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, rank_df


def plot_combined_group_rank_heatmap(
    ranked_df,
    group_order=None,
    variant_order_by_group=None,
    save_path=None,
    figsize=(16, 9),
    title=None,
    show_colorbars=True,
):
    
    """
    Plot one combined heatmap from an already-ranked long table.
    It uses the existing rank_per_group column.
    """

    df = ranked_df.copy()

    required_cols = {"dataset", "group", "variant", "score", "rank_per_group"}

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group order
    if group_order is None:
        group_order = ["single_distribution", "mixture_vs_single", "pareto_vs_baseline", "endo_level"]

    final_group_order = []

    for group in group_order:
        if group in df["group"].unique():
            final_group_order.append(group)

    for group in df["group"].unique():
        if group not in final_group_order:
            final_group_order.append(group)

    group_order = final_group_order

    # Variant order by group
    if variant_order_by_group is None:
        variant_order_by_group = {"mixture_vs_single": ["Baseline", "Normal-only", "Laplace-only", "Student-T-only"],
                                  "single_distribution": ["Normal-only", "Laplace-only", "Student-T-only"],
                                  "pareto_vs_baseline": ["Baseline", "Pareto-std"],
                                  "endo_level": ["No-endo", "Baseline", "Endo-medium", "Endo-large"]}

    # Create wide rank table
    wide_rank = df.pivot_table(index="dataset", columns=["group", "variant"], values="rank_per_group", aggfunc="first")

    ordered_cols = []

    for group in group_order:
        manual_variants = variant_order_by_group.get(group, [])

        existing_manual_variants = []

        for variant in manual_variants:
            if (group, variant) in wide_rank.columns:
                existing_manual_variants.append(variant)

        other_variants = []

        for current_group, variant in wide_rank.columns:
            if current_group == group and variant not in existing_manual_variants:
                other_variants.append(variant)

        for variant in existing_manual_variants + other_variants:
            ordered_cols.append((group, variant))

    wide_rank = wide_rank.loc[:, ordered_cols]

    # Normalize rank for colors
    group_sizes = {}

    for group in df["group"].unique():
        n_variants = df[df["group"] == group]["variant"].nunique()
        group_sizes[group] = n_variants

    wide_color = wide_rank.copy().astype(float)

    for group, variant in wide_color.columns:
        n_variants = group_sizes[group]

        if n_variants > 1:
            wide_color[(group, variant)] = (wide_rank[(group, variant)] - 1) / (n_variants - 1)
        else:
            wide_color[(group, variant)] = 0

    # Colormaps
    group_cmaps = {
        "single_distribution": mpl.colors.LinearSegmentedColormap.from_list(
            "single_distribution_cmap",
            ["#165DA4", "#3E8DC5", "#72B2D7", "#C6DBEF"]),
        "mixture_vs_single": mpl.colors.LinearSegmentedColormap.from_list(
            "mixture_vs_single_cmap",
            ["#17773D", "#2FA051", "#7AC87C", "#C7E9C0"]),
        "pareto_vs_baseline": mpl.colors.LinearSegmentedColormap.from_list(
            "pareto_vs_baseline_cmap",
            ["#9F46D6", "#9C27B0", "#CE93D8", "#F4E9F6"]),
        "endo_level": mpl.colors.LinearSegmentedColormap.from_list(
            "endo_level_cmap",
            ["#DA5512", "#EF7528", "#FDAE6B", "#F5DEC7"])}

    fallback_cmap = mpl.colors.LinearSegmentedColormap.from_list("fallback_cmap", ["#4A5568", "#CBD5E0", "#F7FAFC"])

    # Build RGBA image
    n_rows = wide_rank.shape[0]
    n_cols = wide_rank.shape[1]

    rgba_image = np.ones((n_rows, n_cols, 4))

    for j, (group, variant) in enumerate(wide_rank.columns):
        cmap = group_cmaps.get(group, fallback_cmap)

        for i in range(n_rows):
            color_value = wide_color.iloc[i, j]

            if pd.isna(color_value):
                rgba_image[i, j, :] = (1, 1, 1, 1)
            else:
                rgba_image[i, j, :] = cmap(color_value)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize, dpi=180)

    ax.imshow(rgba_image, aspect="auto")

    x_labels = []

    for group, variant in wide_rank.columns:
        x_labels.append(variant)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=90, ha="right", fontsize=8)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(wide_rank.index, fontsize=7)

    ax.set_title(title, fontsize=12, pad=34)

    # Group labels and separators
    for group in group_order:
        group_cols = []

        for i, (current_group, variant) in enumerate(wide_rank.columns):
            if current_group == group:
                group_cols.append(i)

        if len(group_cols) == 0:
            continue

        start = min(group_cols)
        end = max(group_cols)
        center = (start + end) / 2

        ax.text(center, -1.0, group, ha="center", va="bottom", fontsize=9, fontweight="bold")

        if end < n_cols - 1:
            ax.axvline(end + 0.5, color="white", linewidth=2.5)

    # Annotate raw ranks
    for i in range(n_rows):
        for j in range(n_cols):
            rank_value = wide_rank.iloc[i, j]

            if pd.notna(rank_value):
                ax.text(j, i, str(int(rank_value)), ha="center", va="center", fontsize=5.5, color="#1A202C")

    # Optional colorbars
    if show_colorbars:
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        legend_left = 0.90
        legend_width = 0.015
        legend_height = 0.15
        legend_gap = 0.045
        legend_top = 0.78

        for k, group in enumerate(group_order):
            bottom = legend_top - k * (legend_height + legend_gap)

            cax = fig.add_axes([legend_left, bottom, legend_width, legend_height])

            cmap = group_cmaps.get(group, fallback_cmap)

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["best", "worst"])
            cbar.ax.tick_params(labelsize=6)

            cbar.set_label(group, fontsize=7, rotation=90, labelpad=8)

    else:
        plt.tight_layout()

    # Save
    if save_path is not None:
        save_folder = os.path.dirname(save_path)
        os.makedirs(save_folder, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, wide_rank, wide_color


def sort_results_by_model_order(result_scores: pd.DataFrame) -> pd.DataFrame:
    
    """Sort result scores by a fixed model order"""

    desired_order = [
        "Baseline",
        "Normal-only",
        "Laplace-only",
        "Student-T-only",
        "Pareto-std",
        "No-endo",
        "Endo-medium",
        "Endo-large",
        "Random Forest",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Linear",
    ]

    existing_order = []

    for model in desired_order:
        if model in result_scores["model"].values:
            existing_order.append(model)

    other_models = []

    for model in result_scores["model"].tolist():
        if model not in existing_order:
            other_models.append(model)

    final_order = existing_order + other_models

    result_scores["model"] = pd.Categorical(result_scores["model"], categories=final_order, ordered=True)

    result_scores = result_scores.sort_values("model")
    result_scores = result_scores.reset_index(drop=True)
    result_scores["model"] = result_scores["model"].astype(str)

    return result_scores