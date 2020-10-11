from overrides import overrides
from typing import List, Iterable, Optional

import torch
import torch.distributions

from utils.cache import AttributeValueGaussianCacheEntry

from models.base import EmbeddingModel, EmbeddingDistribution, PyTorchDevice


class GaussianEmbeddingDistribution(EmbeddingDistribution):
    """
    Denotes a general, possibly trainable, pyTorch module that represents an embedding distribution, aka.
    p(h_C|v, a).
    """
    def __init__(self, distribution: torch.distributions.MultivariateNormal):
        self._distribution = distribution
        dimensionality = distribution.mean.shape[0]
        super().__init__(dimensionality)

    @overrides
    def sample(self, num_samples: int) -> torch.Tensor:
        return self._distribution.sample(sample_shape=torch.Size([num_samples]))

    @overrides
    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(input)


class GaussianEmbeddingModel(EmbeddingModel):
    @staticmethod
    def train_from_cache(self):
        return True

    @classmethod
    def from_cache_entries(
            cls, cache_entries: List[AttributeValueGaussianCacheEntry],
            select_dimensions: Optional[Iterable[int]] = None,
            device: Optional[PyTorchDevice] = "cpu"):
        if not select_dimensions:
            dummy_embedding, _ = cache_entries[0].get_gaussian_model_params()
            select_dimensions = list(range(dummy_embedding.shape[0]))
        else:
            select_dimensions = list(select_dimensions)

        dists = []
        for cache_entry in cache_entries:
            mean, cov = cache_entry.get_gaussian_model_params()

            mean = mean[select_dimensions].double().to(device)
            cov = cov[select_dimensions].t()[select_dimensions].t().double().to(device)

            dists.append(GaussianEmbeddingDistribution(
                torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)))

        return cls(embedding_distributions=dists, device=device)
