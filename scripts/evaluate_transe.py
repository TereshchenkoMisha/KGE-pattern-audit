"""Evaluate filtered per-relation validation MRR for a saved TransE checkpoint."""

import csv
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from src.dataset.loader_fb15k_237 import load_fb15k_237
from src.models.transe import TransE


class FilteredEvaluator:
    def __init__(self, model, all_true_triples, num_entities):
        self.model = model
        self.num_entities = num_entities

        self.true_tails = defaultdict(set)
        self.true_heads = defaultdict(set)

        for h, r, t in all_true_triples.tolist():
            # Keep answer sets for efficient head- and tail-query filtering.
            self.true_tails[(h, r)].add(t)
            self.true_heads[(r, t)].add(h)

    @torch.no_grad()
    def evaluate_batch(self, triples, prediction, candidate_batch_size=1024):
        """
        Evaluate a batch of triples for head or tail prediction.
        Returns one filtered rank per input triple.
        """
        device = next(self.model.parameters()).device
        batch_size = len(triples)

        target_triples = torch.tensor(
            triples, dtype=torch.long, device=device
        )

        target_scores = self.model.score_triples(target_triples)

        ranks = torch.zeros(batch_size, dtype=torch.long, device=device)

        for start in range(0, self.num_entities, candidate_batch_size):
            end = min(start + candidate_batch_size, self.num_entities)

            candidates = torch.arange(start, end, device=device)
            num_candidates = len(candidates)

            heads = target_triples[:, 0:1].expand(-1, num_candidates)
            relations = target_triples[:, 1:2].expand(-1, num_candidates)
            tails = target_triples[:, 2:3].expand(-1, num_candidates)

            if prediction == "tail":
                tails = candidates.unsqueeze(0).expand(batch_size, -1)
            elif prediction == "head":
                heads = candidates.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError("prediction must be 'head' or 'tail'")

            candidate_triples = torch.stack(
                (heads, relations, tails), dim=2
            ).reshape(-1, 3)

            scores = self.model.score_triples(candidate_triples)
            scores = scores.reshape(batch_size, num_candidates)

            # Remove other known correct answers while preserving the query target.
            for i, (h, r, t) in enumerate(triples):
                if prediction == "tail":
                    known_answers = self.true_tails[(h, r)]
                    target = t
                else:
                    known_answers = self.true_heads[(r, t)]
                    target = h

                for answer in known_answers:
                    if answer != target and start <= answer < end:
                        scores[i, answer - start] = -float("inf")

            # Optimistic tie handling
            ranks += (
                scores > target_scores.unsqueeze(1)
            ).sum(dim=1)

        return (ranks + 1).cpu().tolist()


def main():
    with open("configs/transe.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    splits = load_fb15k_237(Path(config["dataset_dir"]))

    train = splits["train"].long()
    val = splits["val"].long()

    # Validation filtering masks other true answers known from train and validation.
    all_true = torch.cat([train, val], dim=0)

    
    all_triples = torch.cat(list(splits.values()), dim=0)

    num_entities = int(
        torch.cat((all_triples[:, 0], all_triples[:, 2])).max()
    ) + 1
    num_relations = int(all_triples[:, 1].max()) + 1

    checkpoint = torch.load(
        config["checkpoint_path"],
        map_location=device,
    )

    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config["embedding_dim"],
        p_norm=config["p_norm"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluator = FilteredEvaluator(
        model=model,
        all_true_triples=all_true,
        num_entities=num_entities,
    )

    # Aggregate both prediction directions under each relation ID.
    relation_rr = defaultdict(list)

    validation_triples = val.tolist()
    batch_size = 32

    for start in range(0, len(validation_triples), batch_size):
        batch = validation_triples[start:start + batch_size]

        tail_ranks = evaluator.evaluate_batch(batch, "tail")
        head_ranks = evaluator.evaluate_batch(batch, "head")

        for triple, tail_rank, head_rank in zip(
            batch, tail_ranks, head_ranks
        ):
            h, r, t = triple

            relation_rr[r].append(1.0 / tail_rank)
            relation_rr[r].append(1.0 / head_rank)

        processed = min(start + len(batch), len(validation_triples))
        print(
            f"Processed {processed:,} / "
            f"{len(validation_triples):,} validation triples"
        )

    # Save per-relation validation MRR.
    output_path = Path("relation_mrr_val.csv")

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["relation_id", "mrr", "n_queries"])

        for relation_id in sorted(relation_rr):
            reciprocal_ranks = relation_rr[relation_id]
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

            writer.writerow([
                relation_id,
                f"{mrr:.8f}",
                len(reciprocal_ranks),
            ])

    all_reciprocal_ranks = [
        rr
        for values in relation_rr.values()
        for rr in values
    ]

    global_mrr = sum(all_reciprocal_ranks) / len(
        all_reciprocal_ranks
    )

    expected_queries = 2 * len(validation_triples)
    actual_queries = len(all_reciprocal_ranks)

    print("\nEvaluation completed.")
    print(f"Validation triples: {len(validation_triples):,}")
    print(f"Expected queries: {expected_queries:,}")
    print(f"Actual queries: {actual_queries:,}")
    print(f"Global filtered validation MRR: {global_mrr:.6f}")
    print(f"Relations evaluated: {len(relation_rr)}")
    print(f"Results saved to: {output_path}")

    assert actual_queries == expected_queries, (
        "Query count does not match 2 * validation triples"
    )

    weighted_mrr = sum(
        sum(values) for values in relation_rr.values()
    ) / actual_queries

    assert abs(global_mrr - weighted_mrr) < 1e-12


if __name__ == "__main__":
    main()
