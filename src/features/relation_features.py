from torch import Tensor

from src.dataset.loader_fb15k_237 import DATASET_DIR, load_fb15k_237


def group_train_pairs(train: Tensor) -> dict[int, set[tuple[int, int]]]:
    pairs_by_relation: dict[int, set[tuple[int, int]]] = {}

    for head_id, relation_id, tail_id in train.tolist():
        pairs = pairs_by_relation.setdefault(relation_id, set())
        pairs.add((head_id, tail_id))

    return pairs_by_relation

def get_symmetry_scores(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    inverse_counts: dict[int, int] = {}
    for id, relations in pairs.items():
        inv = 0
        rel = list(relations)
        for (source, destination) in rel:
            if (destination, source) in relations:
                inv+= 1
        inverse_counts[id] = inv / len(relations)
    return inverse_counts

if __name__ == "__main__":
    splits = load_fb15k_237(DATASET_DIR)
    pairs_by_relation = group_train_pairs(splits["train"])
    symmetry_scores = get_symmetry_scores(pairs_by_relation)
    train_counts = {
        relation_id: len(pairs) for relation_id, pairs in pairs_by_relation.items()
    }

    print(f"Relation types: {len(train_counts):,}")
    print(f"Unique train triples: {sum(train_counts.values()):,}")

    for rel in list(pairs_by_relation)[:5]:
        print(f'ID {rel}: train_count={train_counts[rel]} symmetry={symmetry_scores[rel]:.4f}')