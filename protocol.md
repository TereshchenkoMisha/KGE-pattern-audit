# C1 - The pattern audit, at scale

## Research question

Do training-only relation-pattern measurements predict per-relation KGE test MRR for relation types held out from the performance-prediction model, beyond training frequency?

## Data and stages

The study uses the standard PyG train, validation, and test splits of `FB15k_237` and `WordNet18RR`. All relation features are calculated from training triples only.

The pilot is conducted on FB15k-237 with TransE (`seed=0`) and per-relation filtered validation MRR as the target. It uses five relation-level folds shared by both Ridge models. The pilot establishes the feature and evaluation pipeline before the midway check.

At the midway check, the pilot outputs, feature support counts, and composition availability are reviewed. WN18RR analysis begins after this check. WN18RR is used for descriptive exploratory analysis: per-relation feature values, support counts and MRRs are reported, and leave-one-relation-out results are described without claims of reliable meta-prediction, given its 11 relation types. WN18RR does not determine or revise the FB15k-237 confirmatory criterion.

The confirmatory experiment is conducted on FB15k-237 for TransE, DistMult, ComplEx, and RotatE. Each model uses the same predeclared tuning budget, with hyperparameters selected on validation filtered MRR, and five fixed seeds (0-4). The primary target is per-relation mean filtered test MRR across seeds; between-seed variation is reported separately. Standard filtered test MRR covers head- and tail-prediction queries, and the number of test queries is recorded for each relation. Validation MRR is used for the pilot, while test MRR is used only as the confirmatory outcome.

## Feature definitions

Counts use sets of distinct triples or pairs. For target relation `r`, `T_r` denotes its distinct training triples and `P_r = {(h,t): (h,r,t) is in training}` its distinct training pairs.

1. **Symmetry:** the fraction of triples `(h,r,t)` in `T_r` whose reverse `(t,r,h)` is also in training.
2. **Inversion:** the maximum, over `s != r`, of the fraction of pairs `(h,t)` in `P_r` for which `(t,s,h)` is in training. If no other relation has an inverse match, the value is `0`.
3. **Composition:** for each ordered body `(p,q)`, a pair `(h,t)` is body-supported when training contains `(h,p,z)` and `(z,q,t)` for at least one `z`. A body is eligible when it supports at least **10 distinct `(h,t)` pairs**. Its confidence is the fraction of distinct body-supported pairs that also occur in `P_r`; the composition feature is the maximum confidence across eligible bodies. When no body is eligible, composition is recorded as `0` and `composition_body_available=0`; otherwise the flag is `1`. The flag is retained for reporting and is not a predictor in either Ridge model.
4. **1-to-N:** the mean number of distinct tails per distinct head for relation `r` in training.

The frequency predictor is `log(train_count)`, where `train_count = |T_r|` is the number of distinct training triples for the relation.

## Relation eligibility and held-out folds

Relations with fewer than **50 distinct training triples** are excluded as target relations from Ridge analysis, and their IDs and counts are reported. The threshold applies to Ridge target eligibility; descriptive feature values and support counts remain reportable for other relations.

On FB15k-237, eligible relation IDs are shuffled once with a separate fold-assignment seed of `0` and assigned round-robin to five folds for relation-level cross-validation. Each fold holds out its relation types while the Ridge models are fitted on the other four folds. The same fold assignment and held-out relations are used for both Ridge models and, in the confirmatory experiment, for all four KGE outcomes. The fold-to-relation mapping is frozen before the corresponding outcome comparisons and reported, so the held-out relation types are explicit and reproducible. Relations below the 50-triple threshold are excluded from these Ridge evaluations.

WN18RR relation types are deferred until after the midway check. Its descriptive feature and MRR reporting includes per-relation support counts; any leave-one-relation-out Ridge results apply the same 50-triple eligibility threshold and are interpreted as exploratory.

## Pilot models and evaluation

The pilot compares two Ridge regressions on FB15k-237, predicting TransE per-relation filtered validation MRR (`seed=0`):

1. **Frequency baseline:** `log(train_count)`.
2. **Pattern model:** `log(train_count)` plus symmetry, inversion, composition, and 1-to-N.

Predictors are standardized within each training fold, and Ridge regularization is tuned within that fold. Out-of-fold MAE and R-squared are reported for both models, along with paired MAE improvement, defined as baseline MAE minus pattern-model MAE (positive values favor patterns). Both models use identical held-out relations.

## Confirmatory analysis and success criterion

For each of the four KGE models on FB15k-237, the frequency baseline and pattern model predict per-relation mean filtered test MRR across seeds 0-4. Out-of-fold MAE and R-squared, relation-level support and query counts, and paired MAE improvement are reported. Feature ablations remove one pattern feature at a time from the pattern model.

The paired 95% bootstrap confidence interval for MAE improvement is calculated over held-out relations, resampling the same relation types for both models. Improvement is confirmed for a KGE model only when the interval's lower bound is above zero; otherwise the result is inconclusive. R-squared is secondary. Test MRR is not used to select KGE models or features.

## Reporting and reproducibility

The released analysis records the dataset split/version, feature table, composition availability flag, excluded relations, fold-to-relation mapping, model configurations, seeds, filtered MRR query counts, and out-of-fold predictions. Any protocol deviation is documented before the affected evaluation. Feature definitions, thresholds, and the success criterion remain fixed during evaluation.
