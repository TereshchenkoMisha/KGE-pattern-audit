
import random
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.dataset.loader_fb15k_237 import load_fb15k_237
from src.models.transe import TransE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def negative_sampling(
    positive_triples: torch.Tensor,
    num_entities: int,
    num_negatives: int,
) -> torch.Tensor:
    batch_size = positive_triples.size(0)

    negatives = positive_triples.repeat_interleave(
        num_negatives, dim=0
    ).clone()

    random_entities = torch.randint(
        num_entities,
        (batch_size * num_negatives,),
        device=positive_triples.device,
    )

    corrupt_head = torch.rand(
        batch_size * num_negatives,
        device=positive_triples.device,
    ) < 0.5

    negatives[corrupt_head, 0] = random_entities[corrupt_head]
    negatives[~corrupt_head, 2] = random_entities[~corrupt_head]

    return negatives


def main():
    config_path = Path("configs/transe.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset_dir = Path(config["dataset_dir"])
    splits = load_fb15k_237(dataset_dir)

    train_triples = splits["train"].long()

    # Determine the number of entities and relations
    # from the shared ID space of all official splits.
    all_triples = torch.cat(list(splits.values()), dim=0)

    num_entities = int(
        torch.cat((all_triples[:, 0], all_triples[:, 2])).max()
    ) + 1

    num_relations = int(all_triples[:, 1].max()) + 1

    print(f"Training triples: {len(train_triples):,}")
    print(f"Entities: {num_entities:,}")
    print(f"Relations: {num_relations:,}")

    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config["embedding_dim"],
        p_norm=config["p_norm"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    train_triples = train_triples.to(device)

    checkpoint_path = Path(config["checkpoint_path"])
    log_path = Path(config["log_path"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    margin = config["margin"]
    num_negatives = config["num_negatives"]

    for epoch in range(1, num_epochs + 1):
        model.train()

        permutation = torch.randperm(
            train_triples.size(0),
            device=device,
        )

        total_loss = 0.0
        num_batches = 0

        for indices in permutation.split(batch_size):
            positive = train_triples[indices]

            negative = negative_sampling(
                positive,
                num_entities,
                num_negatives,
            )

            positive_scores = model.score_triples(positive)
            negative_scores = model.score_triples(negative)

            negative_scores = negative_scores.view(
                positive.size(0), num_negatives
            )

            loss = F.relu(
                margin
                - positive_scores.unsqueeze(1)
                + negative_scores
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.normalize_entities()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches

        message = (
            f"Epoch {epoch}/{num_epochs}, "
            f"loss={average_loss:.4f}"
        )

        print(message)
        logging.info(message)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "seed": seed,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
