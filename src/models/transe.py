
"""TransE scoring model for head-relation-tail triples."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        p_norm: int = 1,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.p_norm = p_norm

        self.entity_embeddings = nn.Embedding(
            num_entities, embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            num_relations, embedding_dim
        )

        # Initialize embeddings in a range that scales with the dimension.
        bound = 6 / (embedding_dim ** 0.5)

        nn.init.uniform_(
            self.entity_embeddings.weight, -bound, bound
        )
        nn.init.uniform_(
            self.relation_embeddings.weight, -bound, bound
        )

        self.normalize_entities()

    def normalize_entities(self):
        # Keep entity vectors on the unit sphere after initialization and updates.
        with torch.no_grad():
            self.entity_embeddings.weight.copy_(
                F.normalize(
                    self.entity_embeddings.weight,
                    p=2,
                    dim=1,
                )
            )

    def score_triples(self, triples):
        # A lower translation distance means a more plausible triple.
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        distance = torch.norm(
            h + r - t,
            p=self.p_norm,
            dim=1,
        )

        return -distance

    def forward(self, triples):
        return self.score_triples(triples)

