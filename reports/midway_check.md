# C1 - Midway check

## Project question

Do training-only relation-pattern measurements predict per-relation KGE performance on held-out relation types beyond training frequency?

## Pilot status

The Ridge comparison is implemented for five shared relation-level folds. It compares a frequency baseline using `log(train_count)` with a pattern model that adds symmetry, inversion, composition, and 1-to-N. Both models use the same held-out relations. `alpha` is selected within each outer training fold, with standardization inside the cross-validation pipeline.

The FB15k-237 TransE validation table contains per-relation filtered MRR from one seed (`seed=0`). After joining it to the training-only feature table, **222 relations** met both Ridge eligibility conditions out of **237 feature rows**. The 15 exclusions were 14 relations without a validation-MRR row and one relation with fewer than 50 unique training triples. The join retained one row per relation.

The five-fold nested Ridge comparison gave baseline MAE **0.12396**, pattern-model MAE **0.12242**, and `baseline MAE - pattern MAE` **0.00153**. The measured runtime was **4.01 seconds**. This is a pilot comparison on validation MRR for TransE with one seed; the sign is positive, but the pilot does not establish confirmatory improvement.

One training-data composition example is target relation `r=22` (`/education/educational_institution/students_graduates./education/education/student`) and ordered body `(p=81, q=22)`. The body supported **1,672 distinct pairs**, all 1,672 also present under the target relation, for confidence **1.000**. This exceeds the 10-pair minimum support.

## Reproduction

The synthetic check is reproduced from the repository root with:

```powershell
python scripts/compare_ridge.py --synthetic
```

Dataset preparation, feature extraction, TransE training/evaluation, and the real pilot command are documented in [README.md](../README.md). The exact feature table used and the pilot outputs are saved under `results/fb15k237_val_ridge/`; the fold mapping, exclusions, eligibility summary, per-relation predictions, and metrics are retained there. The corresponding validation-MRR input is [relation_mrr_val.csv](../relation_mrr_val.csv).

## Work after the midway check

The next stage is to run TransE, DistMult, ComplEx, and RotatE with five fixed seeds, complete the confirmatory relation-level comparison on FB15k-237, and conduct the descriptive exploratory analysis on WN18RR. The WN18RR analysis follows the midway check and does not determine the FB15k-237 confirmation criterion.
