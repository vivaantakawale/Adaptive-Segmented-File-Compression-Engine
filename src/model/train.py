"""Train a classifier that maps chunk features to best compression algorithm

Loads a labeled dataset, does a stratified train/val/test split, 
trains a HistGradientBoostingClassifier, and reports classification metrics plus metric that actually matters: 
compression ratio on the held-out test set for model_predicted vs always_zstd vs brute force ceiling
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.compressors import registry
from src.features.extract import FEATURE_NAMES

LABEL_COLUMN = "best_algorithm"
DEFAULT_MODEL_PATH = Path("models/algo_selector.joblib")
MIN_EXAMPLES_WARNING = 10


def load_dataset(path: Path) -> pd.DataFrame:
    """Load labeled dataset from disk

    Args:
        path: Path to .parquet or .csv dataset

    Returns:
        loaded dataset
    """
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def stratified_three_way_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test, stratified per-class

    Handles classes with too few examples for sklearn's train_test_split:
    a class of >= 3 gets at least one row in each split, 
    a class of 2 splits between train/test, and a singleton goes entirely to train

    Args:
        df: Labeled dataset to split
        label_col: Column name to stratify by
        val_frac: Target fraction of each class for validation split
        test_frac: Target fraction of each class for test split
        seed: Random seed for per class shuffle

    Returns:
        (train_df, val_df, test_df), each with a fresh 0..n-1 index
    """
    rng = np.random.RandomState(seed)
    train_idx: list = []
    val_idx: list = []
    test_idx: list = []

    for label, group in df.groupby(label_col):
        idx = rng.permutation(group.index.to_numpy())
        n = len(idx)
        if n < MIN_EXAMPLES_WARNING:
            print(f"warning: class {label!r} has only {n} example(s); split for it is not statistically meaningful")

        if n == 1:
            train_idx.extend(idx)
            continue
        if n == 2:
            train_idx.append(idx[0])
            test_idx.append(idx[1])
            continue

        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        while n_test + n_val >= n:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        n_train = n - n_test - n_val

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train : n_train + n_val])
        test_idx.extend(idx[n_train + n_val :])

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def train_model(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = 0,
    class_weight: str | None = None,
) -> HistGradientBoostingClassifier:
    """Train a HistGradientBoostingClassifier on chunk features

    `class_weight="balanced"` reweights loss inversely to class frequency, 
    so a rare algorithm class isn't effectively ignored

    Args:
        x: Training features, columns matching FEATURE_NAMES
        y: Training labels (algorithm names).
        random_state: Seed for classifier's internal randomness
        class_weight: Passed through to HistGradientBoostingClassifier - "balanced" or None

    Returns:
        The fitted classifier
    """
    model = HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight)
    model.fit(x, y)
    return model


def evaluate_classification(model, x: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    """Compute and print classification metrics for model on a split

    Args:
        model: Fitted classifier
        x: Features for split
        y: True labels for split
        split_name: Label used in printed output

    Returns:
        Dict with accuracy, labels, classification_report, and confusion_matrix
    """
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    labels = sorted(set(y) | set(y_pred))
    report = classification_report(y, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=labels)

    print(f"\n--- {split_name} set ({len(y)} chunks) ---")
    print(f"accuracy: {accuracy:.4f}")
    print(classification_report(y, y_pred, labels=labels, zero_division=0))
    print(f"confusion matrix (rows=true, cols=pred), labels={labels}:")
    print(cm)

    return {
        "accuracy": accuracy,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _ratio(total_original: float, total_compressed: float) -> float:
    return total_original / total_compressed if total_compressed else float("inf")


def evaluate_compression_ratio(model, test_df: pd.DataFrame) -> dict:
    """Compute aggregate compression ratio on test chunks

    Compares model_predicted vs always_zstd vs the brute-force ceiling Sizes come from size_<algo> columns rather than recompressing

    Args:
        model: Fitted classifier
        test_df: Test split, including feature columns, size_<algo> columns, and best_size column (from brute force labeling)

    Returns:
        Dict with "model", "always_zstd", and "brute_force_ceiling" ratios
    """
    x_test = test_df[FEATURE_NAMES]
    predictions = model.predict(x_test)

    total_original = test_df["length"].sum()

    model_sizes = np.array(
        [test_df[f"size_{algo}"].iloc[i] for i, algo in enumerate(predictions)]
    )
    total_model = model_sizes.sum()
    total_zstd = test_df["size_zstd"].sum()
    total_brute_force = test_df["best_size"].sum()

    results = {
        "model": _ratio(total_original, total_model),
        "always_zstd": _ratio(total_original, total_zstd),
        "brute_force_ceiling": _ratio(total_original, total_brute_force),
    }

    print("\n--- compression ratio on test set (original_size / compressed_size) ---")
    for name, ratio in results.items():
        print(f"  {name:<20} {ratio:.3f}x")

    return results


def main() -> None:
    """CLI entry point. Parses sys.argv, trains a model, and saves it plus a metrics report"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/labeled/dataset.parquet"))
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    assert set(FEATURE_NAMES).issubset(df.columns), "dataset is missing expected feature columns"

    train_df, val_df, test_df = stratified_three_way_split(
        df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model = train_model(
        train_df[FEATURE_NAMES],
        train_df[LABEL_COLUMN],
        random_state=args.seed,
        class_weight=args.class_weight,
    )

    val_metrics = evaluate_classification(model, val_df[FEATURE_NAMES], val_df[LABEL_COLUMN], "validation")
    test_metrics = evaluate_classification(model, test_df[FEATURE_NAMES], test_df[LABEL_COLUMN], "test")
    ratio_metrics = evaluate_compression_ratio(model, test_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nSaved model to {args.out}")

    report_path = args.out.with_suffix(".metrics.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
                "val": val_metrics,
                "test": test_metrics,
                "compression_ratio": ratio_metrics,
                "registry_algorithms": registry.list_algorithms(),
                "class_weight": args.class_weight,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics report to {report_path}")


if __name__ == "__main__":
    main()
