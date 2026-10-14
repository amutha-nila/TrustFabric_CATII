from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .data import convert_official_liar_archive, download_official_liar_csv, load_csv
from .models import build_baseline, build_optimized, parameter_grid
from .visualize import make_visualizations

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["false", "mixed", "true"]
RANDOM_STATE = 42
BASELINE_NAME = "TF-IDF + Logistic Regression (baseline)"
OPTIMIZED_NAME = "Word + Character TF-IDF + tuned Logistic Regression"


def split_dataset(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    """Preserve official partitions, or create a stratified 70/10/20 split."""
    if "split" in df:
        partitions = {
            name: df.loc[df["split"] == name].reset_index(drop=True)
            for name in ("train", "valid", "test")
        }
        if any(part.empty for part in partitions.values()):
            raise ValueError("A split-aware dataset must contain non-empty train, valid, and test partitions.")
        method = "Official LIAR train/valid/test partitions; exact duplicate statements removed before partitioning."
    else:
        train_and_valid, test = train_test_split(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df.label,
        )
        train, valid = train_test_split(
            train_and_valid,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=train_and_valid.label,
        )
        partitions = {
            "train": train.reset_index(drop=True),
            "valid": valid.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
        method = "Stratified 70/10/20 train/validation/test split; random_state=42."

    for name, part in partitions.items():
        missing = sorted(set(LABELS) - set(part.label))
        if missing:
            raise ValueError(f"The {name} partition is missing required classes: {', '.join(missing)}.")
    return partitions, method


def score_model(
    name: str,
    split: str,
    model,
    x_values: pd.Series,
    y_values: pd.Series,
) -> tuple[dict, np.ndarray, np.ndarray, list[list[int]]]:
    predictions = model.predict(x_values)
    raw_probabilities = model.predict_proba(x_values)
    probabilities = np.zeros((len(x_values), len(LABELS)), dtype=float)
    for column, label in enumerate(model.classes_):
        probabilities[:, LABELS.index(str(label))] = raw_probabilities[:, column]

    class_report = classification_report(
        y_values,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_values,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    one_hot = np.column_stack(
        [(np.asarray(y_values) == label).astype(int) for label in LABELS]
    )
    metrics = {
        "model": name,
        "split": split,
        "accuracy": float(accuracy_score(y_values, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(macro_f1),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_values,
                probabilities,
                labels=LABELS,
                multi_class="ovr",
                average="macro",
            )
        ),
        "pr_auc_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
        "log_loss": float(log_loss(y_values, probabilities, labels=LABELS)),
        "per_class": {
            label: {
                "precision": float(class_report[label]["precision"]),
                "recall": float(class_report[label]["recall"]),
                "f1": float(class_report[label]["f1-score"]),
                "support": int(class_report[label]["support"]),
            }
            for label in LABELS
        },
    }
    matrix = confusion_matrix(y_values, predictions, labels=LABELS).tolist()
    return metrics, predictions, probabilities, matrix


def _cap_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if not max_rows or max_rows >= len(df):
        return df
    group_columns = ["split", "label"] if "split" in df else ["label"]
    fraction = max_rows / len(df)
    groups = [
        group.sample(frac=fraction, random_state=RANDOM_STATE)
        for _, group in df.groupby(group_columns, sort=True, group_keys=False)
    ]
    sampled = pd.concat(groups).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    if sampled.label.nunique() < len(LABELS):
        raise ValueError("Development subset must contain all three classes; increase --max-rows.")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark TrustFabric claim-credibility models.")
    parser.add_argument("--csv", default=None, help="LIAR-style CSV with statement,label columns.")
    parser.add_argument("--text-column", default="statement")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Development-only stratified cap; 0 uses the entire dataset.",
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Development smoke run only; use the default full tuning for final results.",
    )
    args = parser.parse_args()

    raw = Path(args.csv) if args.csv else ROOT / "data/raw/liar.csv"
    if not raw.exists():
        raw = download_official_liar_csv(raw)
    elif args.csv is None:
        archive = raw.with_name("liar_dataset.zip")
        if archive.exists() and "split" not in pd.read_csv(raw, nrows=0).columns:
            raw = convert_official_liar_archive(archive, raw)
    df = load_csv(raw, args.text_column, args.label_column)
    cleaning_summary = dict(df.attrs)
    df = _cap_rows(df, args.max_rows)
    partitions, split_method = split_dataset(df)

    x_train, y_train = partitions["train"].text, partitions["train"].label
    x_valid, y_valid = partitions["valid"].text, partitions["valid"].label
    x_test, y_test = partitions["test"].text, partitions["test"].label
    x_train_valid = pd.concat([x_train, x_valid], ignore_index=True)
    y_train_valid = pd.concat([y_train, y_valid], ignore_index=True)
    records = []
    timings = {}
    search_history = []
    start_time = datetime.now(timezone.utc).isoformat()

    baseline = build_baseline()
    started = time.perf_counter()
    baseline.fit(x_train, y_train)
    timings["baseline_initial_fit_seconds"] = round(time.perf_counter() - started, 3)
    baseline_validation, _, _, _ = score_model(
        BASELINE_NAME, "validation", baseline, x_valid, y_valid
    )
    started = time.perf_counter()
    baseline_final = clone(baseline).fit(x_train_valid, y_train_valid)
    timings["baseline_final_fit_seconds"] = round(time.perf_counter() - started, 3)
    baseline_test, baseline_predictions, _, baseline_matrix = score_model(
        BASELINE_NAME, "test", baseline_final, x_test, y_test
    )
    records.extend([baseline_validation, baseline_test])

    optimized = build_optimized()
    started = time.perf_counter()
    if args.skip_grid_search:
        optimized.fit(x_train, y_train)
        best_params = {"mode": "grid search skipped for development run"}
        best_cv_score = None
        optimized_validation_model = optimized
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            optimized,
            parameter_grid(),
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            verbose=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        optimized_validation_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
        search_history = [
            {
                "rank": int(search.cv_results_["rank_test_score"][index]),
                "mean_test_f1_macro": float(search.cv_results_["mean_test_score"][index]),
                "std_test_f1_macro": float(search.cv_results_["std_test_score"][index]),
                "params": search.cv_results_["params"][index],
            }
            for index in range(len(search.cv_results_["params"]))
        ]
    timings["optimized_search_and_initial_fit_seconds"] = round(
        time.perf_counter() - started, 3
    )
    optimized_validation, _, _, _ = score_model(
        OPTIMIZED_NAME, "validation", optimized_validation_model, x_valid, y_valid
    )

    optimized_final = clone(build_optimized())
    if not args.skip_grid_search:
        optimized_final.set_params(**best_params)
    started = time.perf_counter()
    optimized_final.fit(x_train_valid, y_train_valid)
    timings["optimized_final_fit_seconds"] = round(time.perf_counter() - started, 3)
    optimized_test, optimized_predictions, optimized_probabilities, optimized_matrix = score_model(
        OPTIMIZED_NAME, "test", optimized_final, x_test, y_test
    )
    records.extend([optimized_validation, optimized_test])

    try:
        recorded_dataset_path = raw.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        recorded_dataset_path = raw.name

    output = ROOT / "outputs"
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    dump(optimized_final, output / "models/trustfabric_model.joblib")
    dump(baseline_final, output / "models/trustfabric_baseline.joblib")

    comparison_rows = [
        {key: value for key, value in row.items() if key != "per_class"}
        for row in records
    ]
    pd.DataFrame(comparison_rows).to_csv(
        output / "metrics/model_comparison.csv", index=False
    )
    (output / "metrics/confusion_matrices.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "test": {
                    BASELINE_NAME: baseline_matrix,
                    OPTIMIZED_NAME: optimized_matrix,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reports = {
        BASELINE_NAME: classification_report(
            y_test,
            baseline_predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        OPTIMIZED_NAME: classification_report(
            y_test,
            optimized_predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    (output / "metrics/classification_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    (output / "metrics/classification_report.txt").write_text(
        "BASELINE\n"
        + classification_report(y_test, baseline_predictions, labels=LABELS, zero_division=0)
        + "\nOPTIMIZED\n"
        + classification_report(y_test, optimized_predictions, labels=LABELS, zero_division=0),
        encoding="utf-8",
    )
    (output / "metrics/grid_search_history.json").write_text(
        json.dumps(search_history, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                **row,
                "params": json.dumps(row["params"], sort_keys=True),
            }
            for row in search_history
        ]
    ).to_csv(output / "metrics/grid_search_history.csv", index=False)
    test_frame = partitions["test"].reset_index(drop=True)
    pd.DataFrame(
        {
            "statement_sha256": test_frame.text.map(
                lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
            ),
            "actual": test_frame.label,
            "baseline_predicted": baseline_predictions,
            "baseline_correct": test_frame.label.to_numpy() == baseline_predictions,
            "optimized_predicted": optimized_predictions,
            "optimized_correct": test_frame.label.to_numpy() == optimized_predictions,
        }
    ).to_csv(output / "metrics/error_analysis.csv", index=False)

    benchmark = {
        "project": "TrustFabric",
        "started_at_utc": start_time,
        "dataset": {
            "path": recorded_dataset_path,
            "source": "William Yang Wang, LIAR benchmark (2017), official author-hosted archive.",
            "source_url": "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip",
            "license": "Research use only. Dataset README states original sources retain copyright; no general redistribution license is granted.",
            "original_rows_before_cleaning": cleaning_summary.get("original_rows"),
            "removed_rows": cleaning_summary.get("removed_rows"),
            "duplicate_text_rows": cleaning_summary.get("duplicate_text_rows"),
            "limitations": "Short US political statements; subjective labels aggregated from six classes into three.",
        },
        "dataset_rows": int(len(df)),
        "class_counts": {
            str(label): int(count) for label, count in df.label.value_counts().to_dict().items()
        },
        "class_counts_by_split": {
            name: {
                str(label): int(count)
                for label, count in partition.label.value_counts().to_dict().items()
            }
            for name, partition in partitions.items()
        },
        "split_rows": {name: int(len(partition)) for name, partition in partitions.items()},
        "split_method": split_method,
        "labels": LABELS,
        "scoring": "f1_macro",
        "random_seeds": {
            "split": RANDOM_STATE,
            "model": RANDOM_STATE,
            "cross_validation": RANDOM_STATE,
        },
        "best_cv_f1_macro": best_cv_score,
        "best_params": best_params,
        "timings_seconds": timings,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "results": records,
        "validation_policy": (
            "Hyperparameters selected using training-only three-fold CV. Validation metrics "
            "are reported before refitting; final artifacts are refit on train+validation "
            "and evaluated once on the held-out test partition."
        ),
        "leakage_prevention": (
            "Official source partitions are preserved when available; exact duplicate "
            "statement text is removed before splitting; text vectorizers are fitted "
            "inside each training fold by the sklearn pipeline."
        ),
        "publication_comparison": (
            "Published results are literature context only. No direct numerical comparison "
            "is claimed because this three-class mapping and split differ from many LIAR experiments."
        ),
    }
    (output / "metrics/benchmark.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )

    feature_names = optimized_final.named_steps["features"].get_feature_names_out()
    coefficients = optimized_final.named_steps["classifier"].coef_
    figures = make_visualizations(
        df,
        y_test,
        optimized_predictions,
        optimized_probabilities,
        LABELS,
        records,
        feature_names,
        coefficients,
        output / "figures",
        baseline_predictions=baseline_predictions,
    )
    print(
        json.dumps(
            {
                "test_metrics": [
                    row for row in records if row["split"] == "test"
                ],
                "best_cv_f1_macro": best_cv_score,
                "best_params": best_params,
                "figures": figures,
                "saved_model": str(output / "models/trustfabric_model.joblib"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
