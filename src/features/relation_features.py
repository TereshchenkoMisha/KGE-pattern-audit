from torch import Tensor

from src.dataset.loader_fb15k_237 import DATASET_DIR, load_fb15k_237


def group_train_pairs(train: Tensor) -> dict[int, set[tuple[int, int]]]:
    pairs_by_relation: dict[int, set[tuple[int, int]]] = {}

    for head_id, relation_id, tail_id in train.tolist():
        pairs = pairs_by_relation.setdefault(relation_id, set())
        pairs.add((head_id, tail_id))

    return pairs_by_relation

def get_symmetry_scores(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    symmetry_counts: dict[int, float] = {}
    for id, relations in pairs.items():
        reversed_pairs = {(tail, head) for head, tail in relations}
        symmetry_counts[id] = len(reversed_pairs & relations) / len(relations)
    return symmetry_counts

def get_inversion_scores(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    inverse_counts: dict[int, float] = {}
    for id_a, relations_a in pairs.items():
        reversed_pairs = {(tail, head) for head, tail in relations_a}
        inverse_counts[id_a] = 0.0
        for id_b, relations_b in pairs.items():
            if id_a == id_b:
                continue
            inverse_counts[id_a] = max(inverse_counts[id_a], len(reversed_pairs & relations_b) / len(relations_a))
    return inverse_counts

if __name__ == "__main__":
    splits = load_fb15k_237(DATASET_DIR)

    # Train counts
    pairs_by_relation = group_train_pairs(splits['train'])
    train_counts = {
        relation_id: len(pairs) for relation_id, pairs in pairs_by_relation.items()
    }

    print(f"Relation types: {len(train_counts):,}")
    print(f"Unique train triples: {sum(train_counts.values()):,}")

    # Symmetry
    symmetry_scores = get_symmetry_scores(pairs_by_relation)

    # Inversion
    inversion_scores = get_inversion_scores(pairs_by_relation)

    for rel in list(pairs_by_relation)[:5]:
        print(f'ID {rel}: train_count={train_counts[rel]} symmetry={symmetry_scores[rel]:.4f} inversion={inversion_scores[rel]:.4f}')