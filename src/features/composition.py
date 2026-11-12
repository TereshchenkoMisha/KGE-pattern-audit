from dataclasses import dataclass
from pathlib import Path
from src.dataset.loader_fb15k_237 import DATASET_DIR, load_fb15k_237
from src.features.relation_features import group_train_pairs
import csv

@dataclass
class CompositionResult:
    scores: dict[int, float]
    body_available: bool
    best_bodies: dict[int, tuple[tuple[int, int] , int, int]]


def build_adjacency(
    pairs_by_relation: dict[int, set[tuple[int, int]]],
) -> tuple[dict[int, dict[int, set[int]]], dict[int, dict[int, set[int]]]]:
    """Index incoming and outgoing edges by node and relation"""
    incoming: dict[int, dict[int, set[int]]] = {}
    outcoming: dict[int, dict[int, set[int]]] = {}
    for id, relations in pairs_by_relation.items():
        for (head, tail) in relations:
            incoming.setdefault(tail, {}).setdefault(id, set()).add(head)
            outcoming.setdefault(head, {}).setdefault(id, set()).add(tail)
    return (incoming, outcoming)


def find_candidate_bodies(
    incoming: dict[int, dict[int, set[int]]],
    outgoing: dict[int, dict[int, set[int]]],
    min_support: int,
) -> dict[tuple[int, int] , list[int]]:
    """Keep bodies that have at least min_support paths before deduplication"""
    path_counts: dict[tuple[int, int], int] = {}
    middle_nodes: dict[tuple[int, int], list[int]] = {}

    for middle in incoming.keys() & outgoing.keys():
        for p, heads in incoming[middle].items():
            for q, tails in outgoing[middle].items():
                body = (p, q)
                new_paths = len(heads) * len(tails)
                path_counts[body] = path_counts.get(body, 0) + new_paths
                middle_nodes.setdefault(body, []).append(middle)

    return {
        body: nodes # body is pair of two relations, nodes is a list of points through which these realtions/edges pass
        for body, nodes in middle_nodes.items()
        if path_counts[body] >= min_support
    }


def collect_supported_pairs(
    body: tuple[int, int], # p, q - relations
    middle_nodes: list[int],
    incoming: dict[int, dict[int, set[int]]],
    outgoing: dict[int, dict[int, set[int]]],
) -> set[tuple[int, int]]:
    """Join edges at the middle nodes and deduplicate their endpoint pairs"""
    p, q = body
    supported_pairs = set()

    for middle in middle_nodes:
        heads = incoming[middle][p]
        tails = outgoing[middle][q]
        supported_pairs.update((head, tail) for head in heads for tail in tails)
    return supported_pairs


def composition_scores(
    pairs_by_relation: dict[int, set[tuple[int, int]]],
    min_support: int = 10,
) -> CompositionResult:
    """Find the best composition confidence for each target relation"""
    incoming, outgoing = build_adjacency(pairs_by_relation)
    candidates = find_candidate_bodies(incoming, outgoing, min_support)

    relations_by_pair: dict[tuple[int, int], list[int]] = {}
    for relation_id, pairs in pairs_by_relation.items():
        for pair in pairs:
            relations_by_pair.setdefault(pair, []).append(relation_id)

    scores = {relation_id: 0.0 for relation_id in pairs_by_relation}
    best_bodies: dict[int, tuple[tuple[int, int], int, int]] = {}
    body_available = False

    for body, middle_nodes in sorted(candidates.items()):
        supported_pairs = collect_supported_pairs(body, middle_nodes, incoming, outgoing)
        support = len(supported_pairs)

        if support >= min_support:
            body_available = True
            matches_by_relation: dict[int, int] = {}

            for pair in supported_pairs:
                for relation_id in relations_by_pair.get(pair, ()):
                    matches_by_relation[relation_id] = (matches_by_relation.get(relation_id, 0) + 1)

            for relation_id, matches in matches_by_relation.items():
                confidence = matches / support
                if confidence > scores[relation_id]:
                    scores[relation_id] = confidence
                    best_bodies[relation_id] = (body, matches, support)

        del supported_pairs

    return CompositionResult(
        scores=scores,
        body_available=body_available,
        best_bodies=best_bodies,
    )


def write_feature_table(
    base_path: Path,
    output_path: Path,
    result: CompositionResult,
) -> None:
    """Прочитать базовый CSV и добавить две колонки по relation_id."""
    with base_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if fieldnames is None or "relation_id" not in fieldnames:
            raise ValueError(f"Missing relation_id column in {base_path}")
        rows = list(reader)

    relation_ids = [int(row["relation_id"]) for row in rows]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError(f"Duplicate relation IDs in {base_path}")
    if set(relation_ids) != set(result.scores):
        raise ValueError("Base features and composition scores have different relation IDs")

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[*fieldnames, "composition", "composition_body_available"],
        )
        writer.writeheader()

        for row, relation_id in zip(rows, relation_ids, strict=True):
            writer.writerow({
                **row,
                "composition": result.scores[relation_id],
                "composition_body_available": int(result.body_available),
            })

def main() -> None:
    splits = load_fb15k_237(DATASET_DIR)
    pairs_by_relation = group_train_pairs(splits["train"])
    result = composition_scores(pairs_by_relation)

    write_feature_table(
        base_path=DATASET_DIR / "relation_features_base.csv",
        output_path=DATASET_DIR / "relation_features.csv",
        result=result,
    )

    print(f"Composition results saved to {DATASET_DIR / "relation_features.csv"}")


if __name__ == "__main__":
    main()