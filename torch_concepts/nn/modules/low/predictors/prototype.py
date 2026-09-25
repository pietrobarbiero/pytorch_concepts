import torch
from typing import Optional, Union
import torch.nn.functional as F

from .....annotations import Annotations
from ..base .layer import BaseConceptLayer
from ..ops import StraightThroughSoftmax


class PrototypeEmbeddingToSimilarity(BaseConceptLayer):
    def __init__(
        self,
        n_features: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        temperature: float = 1.0,
        temp_forward: Optional[float] = None,
        use_straight_through: bool = True,
    ):
        # Infer in_embeddings
        in_embeddings = n_features

        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,  # Concepts come from encoder, not traditional input
            in_embeddings=in_embeddings
        )
        self.n_features = n_features

        # Temperature settings
        self.use_straight_through = use_straight_through
        if use_straight_through:
            self.temp_forward = temp_forward if temp_forward is not None else 0.01
            self.temp_backward = temperature
        else:
            self.temp_forward = temperature
            self.temp_backward = temperature

        self.register_buffer('temperature_forward', torch.tensor(self.temp_forward))
        self.register_buffer('temperature_backward', torch.tensor(self.temp_backward))


    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = embeddings.shape[0]

        # Compute similarities between embeddings and prototypes
        # embeddings: [batch, n_features] -> [batch, 1, 1, n_features]
        # concepts: [batch_proto, num_concepts, n_features] -> [1, batch_proto, num_concepts, n_features]
        x_expanded = embeddings.view(batch_size, 1, 1, embeddings.shape[-1])
        proto_expanded = concepts.permute(1, 0, 2).unsqueeze(0)

        # Compute squared distances: [batch, num_concepts, max_prototypes]
        squared_distances = torch.sum((x_expanded - proto_expanded) ** 2, dim=-1)

        # Use negative squared distances as logits
        logits = -squared_distances

        # Compute similarities with appropriate method
        # similarity shape: [batch, num_concepts, batch_proto]
        if self.use_straight_through:
            # Straight-through: peaked forward, smooth backward
            similarity = StraightThroughSoftmax.apply(
                logits,
                self.temperature_forward,
                self.temperature_backward,
                2  # dim for softmax
            )
        else:
            # Standard softmax with single temperature
            temp = torch.clamp(self.temperature_forward, min=0.1)
            similarity = F.softmax(logits / temp, dim=2)

        return similarity


class ConceptSimilarityToConcept(BaseConceptLayer):
    """
    Aggregates concept activations from prototypes to final concept predictions using similarity scores.

    Args:
        out_concepts: Number of output concepts (num_concepts).
    """
    def __init__(
        self,
        out_concepts: Union[int, Annotations],
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,
            in_embeddings=None
        )

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate concept weights with embeddings via prototype similarity.

        Args:
            concepts: Tensor of shape [max_prototypes, num_concepts] - concept activations for prototypes.
            embeddings: Tensor of shape [batch, num_concepts, max_prototypes] - similarity scores between embeddings and prototypes.

        Returns:
            torch.Tensor: Tensor of shape [batch, num_concepts] - final concept predictions.
        """
        # Weighted sum over prototypes
        # embeddings: [batch, num_concepts, max_prototypes]
        # concepts: [max_prototypes, num_concepts]
        output = torch.einsum('bcm,mc->bc', embeddings, concepts)

        return output
