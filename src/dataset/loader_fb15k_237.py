import csv
from pathlib import Path
import torch
from torch_geometric.datasets import FB15k_237


RAW_FILES = {'train': 'train.txt', 'val': 'valid.txt', 'test': 'test.txt'}
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'
DATASET_DIR = DATA_DIR / 'FB15k-237'

def load_fb15k_237(root: str | Path) -> dict[str, torch.Tensor]:
    """Return the official splits as tensors with columns (head, relation, tail)"""
    splits = {}
    for split in RAW_FILES:
        data = FB15k_237(root=str(root), split=split)[0]
        heads, tails = data.edge_index # get IDs of heads and tails
        splits[split] = torch.stack((heads, data.edge_type, tails), dim=1) # data.edge_type contains relation IDs
        # torch.stack creates the tensors of the fprm: [[head_id, relation_id, tail_id],]
    return splits


def save_relation_names(root: Path, splits: dict[str, torch.Tensor]) -> None:
    """Match relation IDs to names using the original rows in PyG's cache"""
    raw_dir = root / 'raw'
    if not all((raw_dir / filename).is_file() for filename in RAW_FILES.values()):
        print("Raw split files are unavailable... Skipping relations.tsv")
        return

    names: dict[int, str] = {}
    for split, filename in RAW_FILES.items():
        relation_ids = splits[split][:, 1].tolist()

        with (raw_dir / filename).open(encoding='utf-8', newline='') as source:
            raw_triples = csv.reader(source, delimiter='\t')

            for relation_id, (_, name, _) in zip(relation_ids, raw_triples, strict=True):
                if names.setdefault(relation_id, name) != name:
                    raise ValueError(f"Inconsistent name for relation ID {relation_id}")

    output = root / 'relations.сsv'
    with output.open('w', encoding='utf-8', newline='') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(('relation_id', 'relation_name'))
        writer.writerows(sorted(names.items()))
    print(f"Relation names saved to {output}")


def main() -> None:
    splits = load_fb15k_237(DATASET_DIR)

    print()
    print("------------------------------ FB15k-237 dataset statistics ------------------------------")
    for split, triples in splits.items():
        num_relations = triples[:, 1].unique().numel()
        print(f'{split}: {len(triples):,} triples, {num_relations} relation types')

    all_relation_ids = torch.cat([triples[:, 1] for triples in splits.values()])
    print(f'\nTotal relation types: {all_relation_ids.unique().numel()}\n')
    save_relation_names(DATASET_DIR, splits)


if __name__ == "__main__":
    main()
