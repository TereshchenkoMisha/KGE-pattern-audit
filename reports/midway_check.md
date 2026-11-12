# C1 - Midway check

Project update | Week 9

Owner: Anna | Review: Mikhail and Saida

## Project question

Do training-only relation-pattern measurements predict per-relation KGE performance on held-out relation types beyond training frequency?

## Pilot status

The Ridge comparison is implemented for five shared relation-level folds. It compares a frequency baseline using `log(train_count)` with a pattern model that adds symmetry, inversion, composition, and 1-to-N. Both models use the same held-out relations. `alpha` is selected within each outer training fold, with standardization inside the cross-validation pipeline.

The synthetic pipeline check joined **25 feature rows** to **25 validation rows** without changing the row count. Every relation received exactly one out-of-fold prediction from each model, and both MAEs were calculated on the same 25 rows. The synthetic comparison gave baseline MAE **0.0518**, pattern-model MAE **0.0383**, and `baseline MAE - pattern MAE` **0.0135**. The recorded runtime was **5.78 seconds**. This is a pipeline check only; it is not evidence about FB15k-237.

The FB15k-237 validation pilot is **pending**. The required `relation_features.csv` and `relation_mrr_val.csv` are not present in the project workspace. Therefore the real number of eligible relations, real MAEs, and elapsed pilot time cannot yet be reported. The comparison script will report relation counts before and after eligibility filters, exclusion reasons, out-of-fold predictions, fold assignment, both MAEs, their difference, and runtime when run with those files.

The composition rule's minimum-support example is illustrative, not a measured dataset example: if an ordered body supports 12 distinct `(h,t)` pairs and 8 also occur under the target relation, it passes the 10-pair threshold and has confidence `8/12 = 0.667`. A real example and its support count remain pending the feature table.

## Reproduction

The synthetic pipeline check is reproduced from the repository root with:

```powershell
python scripts/compare_ridge.py --synthetic
```

The real pilot command and CSV column requirements are documented in [README.md](../README.md). Its results are saved under `results/fb15k237_val_ridge/` by default.

## Work after the midway check

The next stage is to run TransE, DistMult, ComplEx, and RotatE with five fixed seeds, complete the confirmatory relation-level comparison on FB15k-237, and conduct the descriptive exploratory analysis on WN18RR. The WN18RR analysis follows the midway check and does not determine the FB15k-237 confirmation criterion.
