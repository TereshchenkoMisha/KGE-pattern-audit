"""Nested, paired Ridge comparison on held-out relation types."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["symmetry", "inversion", "composition", "one_to_n"]
NUMERIC = ["train_count", *FEATURES, "filtered_mrr", "validation_queries"]
ALPHAS = np.logspace(-4, 4, 33)


def load_inputs(features_path: Path, mrr_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_path)
    mrr = pd.read_csv(mrr_path)
    required_features = {"relation_id", "train_count", *FEATURES}
    required_mrr = {"relation_id", "filtered_mrr", "validation_queries"}
    if missing := required_features - set(features.columns):
        raise ValueError(f"relation_features.csv is missing columns: {sorted(missing)}")
    if missing := required_mrr - set(mrr.columns):
        raise ValueError(f"relation_mrr_val.csv is missing columns: {sorted(missing)}")
    features = features[["relation_id", "train_count", *FEATURES]]
    mrr = mrr[["relation_id", "filtered_mrr", "validation_queries"]]
    for name, frame in (("relation_features.csv", features), ("relation_mrr_val.csv", mrr)):
        duplicate_ids = frame.loc[frame["relation_id"].duplicated(keep=False), "relation_id"]
        if not duplicate_ids.empty:
            raise ValueError(
                f"{name} contains duplicate relation IDs; join would multiply rows: "
                f"{sorted(duplicate_ids.astype(str).unique())}"
            )

    joined = features.merge(
        mrr, how="outer", on="relation_id", validate="one_to_one", indicator=True
    )
    joined[NUMERIC] = joined[NUMERIC].apply(pd.to_numeric, errors="coerce")
    joined["exclusion_reason"] = ""

    def add_reason(mask: pd.Series, reason: str) -> None:
        prior = joined.loc[mask, "exclusion_reason"]
        joined.loc[mask, "exclusion_reason"] = np.where(prior.eq(""), reason, prior + ";" + reason)

    add_reason(joined["_merge"].eq("left_only"), "missing_validation_mrr_row")
    add_reason(joined["_merge"].eq("right_only"), "missing_relation_features_row")
    add_reason(joined["train_count"].notna() & joined["train_count"].lt(50), "fewer_than_50_unique_train_triples")
    add_reason(
        joined["validation_queries"].notna() & joined["validation_queries"].lt(1),
        "no_validation_queries",
    )
    bad_values = joined[NUMERIC].isna().any(axis=1) & joined["_merge"].eq("both")
    add_reason(bad_values, "missing_or_non_numeric_required_value")
    eligible = joined.loc[joined["exclusion_reason"].eq("")].copy()
    if (eligible["train_count"] <= 0).any():
        raise ValueError("Eligible train_count values must be positive")
    if not eligible["filtered_mrr"].between(0, 1).all():
        raise ValueError("filtered_mrr values must be between 0 and 1")
    return joined.drop(columns="_merge"), eligible


def synthetic_input() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    n = 25
    count = rng.integers(60, 900, n)
    pattern = rng.uniform(0, 1, (n, 3))
    one_to_n = rng.uniform(1, 7, n)
    log_count = np.log(count)
    mrr = np.clip(
        0.38 - 0.025 * (log_count - log_count.mean()) + 0.09 * pattern[:, 0]
        + 0.05 * pattern[:, 1] + 0.08 * pattern[:, 2] - 0.015 * (one_to_n - 4)
        + rng.normal(0, 0.035, n),
        0.03,
        0.95,
    )
    return pd.DataFrame(
        {
            "relation_id": [f"synthetic_relation_{i:02d}" for i in range(n)],
            "train_count": count,
            "symmetry": pattern[:, 0],
            "inversion": pattern[:, 1],
            "composition": pattern[:, 2],
            "one_to_n": one_to_n,
            "filtered_mrr": mrr,
            "validation_queries": rng.integers(10, 300, n),
        }
    )


def _predict_nested(
    x: np.ndarray, y: np.ndarray, outer_splits: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[float]]:
    predictions = np.full(len(y), np.nan)
    alphas: list[float] = []
    for train, held_out in outer_splits:
        inner_k = min(5, len(train))
        if inner_k < 2:
            raise ValueError("Outer training folds need at least two relations for alpha tuning")
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        search = GridSearchCV(
            model,
            {"ridge__alpha": ALPHAS},
            scoring="neg_mean_absolute_error",
            cv=KFold(n_splits=inner_k, shuffle=True, random_state=0),
        )
        search.fit(x[train], y[train])
        predictions[held_out] = search.predict(x[held_out])
        alphas.append(float(search.best_params_["ridge__alpha"]))
    if not np.isfinite(predictions).all():
        raise AssertionError("Every relation must have exactly one out-of-fold prediction")
    return predictions, alphas


def compare(data: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = data.sort_values("relation_id", kind="stable").reset_index(drop=True)
    n = len(data)
    if n < 10:
        raise ValueError(f"Need at least 10 eligible relations for five-fold nested CV; got {n}")
    y = data["filtered_mrr"].to_numpy(float)
    frequency = np.log(data[["train_count"]].to_numpy(float))
    pattern = np.column_stack([frequency, data[FEATURES].to_numpy(float)])
    shuffled = np.random.RandomState(seed).permutation(n)
    fold_id = np.empty(n, dtype=int)
    fold_id[shuffled] = np.arange(n) % 5
    outer = [(np.flatnonzero(fold_id != f), np.flatnonzero(fold_id == f)) for f in range(5)]
    baseline_pred, baseline_alpha = _predict_nested(frequency, y, outer)
    pattern_pred, pattern_alpha = _predict_nested(pattern, y, outer)

    predictions = data[["relation_id", "filtered_mrr"]].rename(columns={"filtered_mrr": "true_mrr"})
    predictions.insert(1, "fold", fold_id + 1)
    predictions["baseline_prediction"] = baseline_pred
    predictions["pattern_prediction"] = pattern_pred
    if predictions["relation_id"].duplicated().any():
        raise AssertionError("Each relation must occur once in the prediction table")
    if predictions[["true_mrr", "baseline_prediction", "pattern_prediction"]].isna().any().any():
        raise AssertionError("Both models must predict every row with a target")

    baseline_mae = float(mean_absolute_error(y, baseline_pred))
    pattern_mae = float(mean_absolute_error(y, pattern_pred))
    metrics = {
        "n_relations": n,
        "baseline_mae": baseline_mae,
        "pattern_mae": pattern_mae,
        "mae_improvement": baseline_mae - pattern_mae,
    }
    folds = pd.DataFrame(
        {
            "fold": range(1, 6),
            "held_out_relations": [
                ";".join(predictions.loc[predictions["fold"].eq(f), "relation_id"].astype(str))
                for f in range(1, 6)
            ],
            "baseline_alpha": baseline_alpha,
            "pattern_alpha": pattern_alpha,
        }
    )
    return predictions, folds, metrics


def run_synthetic(output_dir: Path) -> dict:
    started = time.perf_counter()
    source = synthetic_input()
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        source[["relation_id", "train_count", *FEATURES]].to_csv(tmp / "relation_features.csv", index=False)
        source[["relation_id", "filtered_mrr", "validation_queries"]].to_csv(
            tmp / "relation_mrr_val.csv", index=False
        )
        joined, eligible = load_inputs(tmp / "relation_features.csv", tmp / "relation_mrr_val.csv")
    if len(joined) != len(source) or len(eligible) != len(source):
        raise AssertionError("Synthetic one-to-one join changed the row count")
    predictions, folds, metrics = compare(eligible)
    common = predictions.dropna(subset=["true_mrr", "baseline_prediction", "pattern_prediction"])
    if len(common) != metrics["n_relations"]:
        raise AssertionError("Both MAEs must be calculated on the same relation rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    folds.to_csv(output_dir / "folds.csv", index=False)
    metrics.update(
        {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "fold_seed": 0,
            "joined_relations": int(len(joined)),
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_real(features: Path, mrr: Path, output_dir: Path, seed: int) -> dict:
    started = time.perf_counter()
    joined, eligible = load_inputs(features, mrr)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.loc[joined["exclusion_reason"].ne("")].to_csv(output_dir / "exclusions.csv", index=False)
    eligible.to_csv(output_dir / "eligible_relations.csv", index=False)
    counts = {
        "n_relations_before_filter": int(len(joined)),
        "n_relations_after_filter": int(len(eligible)),
        "n_excluded": int(len(joined) - len(eligible)),
    }
    (output_dir / "eligibility.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    predictions, folds, metrics = compare(eligible, seed)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    folds.to_csv(output_dir / "folds.csv", index=False)
    metrics.update({**counts, "runtime_seconds": round(time.perf_counter() - started, 3), "fold_seed": seed})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--mrr", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fb15k237_val_ridge"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        result = run_synthetic(Path("results/synthetic_ridge"))
    else:
        if args.features is None or args.mrr is None:
            parser.error("--features and --mrr are required unless --synthetic is set")
        result = run_real(args.features, args.mrr, args.output_dir, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
