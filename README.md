# KGE-pattern-audit

## Prepare the FB15k-237 dataset

From the repository root, create a Python 3.12 environment and install the dependencies:

```bash
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

If you already have the `venv` environment, start with `source venv/bin/activate`.

Run the dataset loader:

```bash
python src/dataset/loader_fb15k_237.py
```

On the first run, PyTorch Geometric downloads and processes the official train,
validation, and test splits under `data/FB15k-237/`. Later runs reuse the local
cache.

The loader prints some dataset statistics: the number of triples and relation types in each split and
saves a tab-separated table that maps relation IDs to their original names

## Feature counting

To obtain statistics on graph relations execute the following command

```bash
python -m src.features.relation_features
```
