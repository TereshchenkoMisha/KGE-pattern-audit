# KGE-pattern-audit

## Ridge comparison on held-out relation types

The Ridge pilot compares a frequency baseline (`log(train_count)`) with the same predictor plus `symmetry`, `inversion`, `composition`, and `one_to_n`. Five shared outer folds hold out complete relation types. Each model's `alpha` is selected only within the outer training portion; `StandardScaler` is part of the pipeline and is refit in every inner training split.

### Synthetic pipeline check

From the repository root, run:

```powershell
python -m pip install numpy pandas scikit-learn
python scripts/compare_ridge.py --synthetic
```

This writes `results/synthetic_ridge/predictions.csv`, `folds.csv`, and `metrics.json`. The deterministic synthetic example checks that each relation has exactly one prediction from each model and both MAEs use the same rows. It is a pipeline check, not a project result.

### FB15k-237 validation pilot

The real pilot expects one row per relation in each CSV:

- `relation_features.csv`: `relation_id`, `train_count`, `symmetry`, `inversion`, `composition`, `one_to_n`.
- `relation_mrr_val.csv`: `relation_id`, `filtered_mrr`, `validation_queries`.

This repository snapshot contains the comparison script but not the feature-extraction or TransE training/evaluation scripts from dependencies 04, 07, and 08. The command below starts from their prepared CSV outputs; those upstream commands need to be added when the dependencies are brought into this branch:

```powershell
python scripts/compare_ridge.py --features path/to/relation_features.csv --mrr path/to/relation_mrr_val.csv
```

An optional `--output-dir` changes the default output folder `results/fb15k237_val_ridge`; `--seed` sets the outer-fold seed and defaults to `0`. The script rejects duplicate relation IDs before joining, uses a one-to-one outer join, and records exclusions for missing rows/values, fewer than 50 unique train triples, or no validation queries. It writes `eligibility.json`, `exclusions.csv`, `eligible_relations.csv`, `predictions.csv`, `folds.csv`, and `metrics.json`. Eligibility counts and reasons are saved before model fitting, including when too few relations remain for cross-validation. Each prediction row contains the fold, true validation MRR, and both predictions. The metrics include the two MAEs, `baseline_mae - pattern_mae`, included/excluded counts, and elapsed time.

### Midway check and work after it

The short midway report is in `reports/midway_check.md`. The real FB15k-237 pilot requires the two input CSVs above; the current report marks its results and data-derived composition example as pending rather than substituting synthetic values. After the midway check, the study proceeds to four KGE models and five seeds, the confirmatory FB15k-237 comparison, and descriptive WN18RR analysis.
