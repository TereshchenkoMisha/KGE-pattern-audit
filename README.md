# KGE-pattern-audit

This project tests whether training-only relation patterns predict per-relation knowledge graph embedding performance beyond training frequency. Feature definitions and evaluation choices are recorded in [protocol.md](protocol.md).

## Environment and data

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Load the standard PyG FB15k-237 splits (the first run downloads and processes the dataset):

```bash
python src/dataset/loader_fb15k_237.py
```

## Extract relation features

Calculate train counts, symmetry, inversion, and mean tails per head:

```bash
python -m src.features.relation_features
```

Then add composition and its availability flag:

```bash
python -m src.features.composition
```

The complete table is written to `data/FB15k-237/relation_features.csv`. It includes the best body's relation IDs, matched-pair count, distinct-pair support, and composition availability for audit.

## Train and evaluate TransE

Train the configured TransE model and save its checkpoint:

```bash
python scripts/train_transe.py
```

Evaluate filtered validation MRR by relation using the saved checkpoint:

```bash
python scripts/evaluate_transe.py
```

This writes `relation_mrr_val.csv` in the project root.

## Ridge comparison on held-out relations

The comparison uses five shared relation-level folds. The frequency baseline uses `log(train_count)`; the pattern model adds symmetry, inversion, composition, and 1-to-N. Alpha is tuned within each outer training fold, with `StandardScaler` in the pipeline so scaling is refit inside each inner fold.

Run the deterministic synthetic pipeline check:

```bash
python scripts/compare_ridge.py --synthetic
```

Run the FB15k-237 validation pilot after extracting the feature and TransE MRR tables:

```bash
python scripts/compare_ridge.py --features data/FB15k-237/relation_features.csv --mrr relation_mrr_val.csv
```

The script accepts the feature column names `one_to_n` or `tails_per_head` and MRR columns `filtered_mrr` or `mrr`, `validation_queries` or `n_queries`. It checks for unique relation IDs before the one-to-one join. Ridge eligibility requires at least 50 unique training triples and one or more validation queries. Outputs under `results/fb15k237_val_ridge/` include a snapshot of the feature table used, eligibility counts, exclusion reasons, the fold-to-relation mapping, per-relation true MRR and both predictions, and the paired MAEs.

The midway report is in [reports/midway_check.md](reports/midway_check.md). After the check, the next stages are four KGE models with five seeds, the confirmatory FB15k-237 comparison, and descriptive WN18RR analysis.
