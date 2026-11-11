import csv
from torch import Tensor
from src.dataset.loader_fb15k_237 import DATASET_DIR, load_fb15k_237


def group_train_pairs(train: Tensor) -> dict[int, set[tuple[int, int]]]:
    pairs_by_relation: dict[int, set[tuple[int, int]]] = {}

    for head_id, relation_id, tail_id in train.tolist():
        pairs = pairs_by_relation.setdefault(relation_id, set())
        pairs.add((head_id, tail_id))

    return pairs_by_relation

def tails_per_head(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for id, relations in pairs.items():
        scores[id] = len(relations) / len({head for (head, _) in relations})
    return scores

def symmetry(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for id, relations in pairs.items():
        reversed_pairs = {(tail, head) for head, tail in relations}
        scores[id] = len(reversed_pairs & relations) / len(relations)
    return scores

def inversion(pairs: dict[int, set[tuple[int, int]]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for id_a, relations_a in pairs.items():
        reversed_pairs = {(tail, head) for head, tail in relations_a}
        scores[id_a] = 0.0
        for id_b, relations_b in pairs.items():
            if id_a == id_b:
                continue
            scores[id_a] = max(scores[id_a], len(reversed_pairs & relations_b) / len(relations_a))
    return scores

def main() -> None:
    splits = load_fb15k_237(DATASET_DIR)

    # Train counts
    pairs_by_relation = group_train_pairs(splits['train'])
    train_counts = {
        relation_id: len(pairs) for relation_id, pairs in pairs_by_relation.items()
    }

    print(f"Relation types: {len(train_counts):,}")
    print(f"Unique train triples: {sum(train_counts.values()):,}")

    # Symmetry
    symmetry_scores = symmetry(pairs_by_relation)

    # Inversion
    inversion_scores = inversion(pairs_by_relation)

    # Tails per head - the average number of tail entities connected to a given head entity for a specific relation
    tails_per_head_scores = tails_per_head(pairs_by_relation)

    # Statistics for checking

    # for rel in list(pairs_by_relation)[:5]:
    #     print(f'ID {rel}: train_count={train_counts[rel]} \t symmetry={symmetry_scores[rel]:.4f} \t inversion={inversion_scores[rel]:.4f}\t'\
    #           f'tails_per_head={tails_per_head_scores[rel]:.4f}')

    # Store to csv file
    output = DATASET_DIR / 'relation_features_base.csv'

    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['relation_id', 'train_count', 'symmetry', 'inversion', 'tails_per_head'])

        for relation_id in sorted(train_counts):
            writer.writerow([
                relation_id,
                train_counts[relation_id],
                symmetry_scores[relation_id],
                inversion_scores[relation_id],
                tails_per_head_scores[relation_id],
            ])

    print(f'Saved relation features to {output}')

if __name__ == "__main__":
    main()