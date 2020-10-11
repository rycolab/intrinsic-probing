from typing import List, Iterable, Optional, Tuple, Union
from math import isclose

import torch
import torch.distributions
import torch.nn as nn

from utils.cache import AttributeValueGaussianCacheEntry

from readers.base import Reader
from utils.word import Word


PyTorchDevice = Union[torch.device, str]


class EmbeddingDistribution(nn.Module):
    """
    Denotes a general, possibly trainable, pyTorch module that represents an embedding distribution, aka.
    p(h_C|v, a).
    """
    def __init__(self, dimensionality: int):
        self._dimensionality = dimensionality

        super().__init__()

    def sample(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_dimensionality(self) -> int:
        return self._dimensionality


class ValueModel:
    def __init__(self, value_probs: List[Tuple[str, float]], device: PyTorchDevice = "cpu"):
        self._values, self._probs = zip(*value_probs)
        self._distribution = torch.distributions.Categorical(torch.tensor(self._probs).double().to(device))
        self._device = device

        self._values_dict = {val: idx for idx, val in enumerate(self._values)}

    @classmethod
    def create_uniform(cls, values: List[str], device: PyTorchDevice = "cpu"):
        """
        Creates a attribute-value model that assigns uniform probability to each possibility.
        """
        probs = [1.0 / float(len(values))] * len(values)
        return cls(list(zip(values, probs)), device=device)

    @classmethod
    def from_cache_entries(
            cls, cache_entries: Iterable[AttributeValueGaussianCacheEntry], device: PyTorchDevice = "cpu"):
        """
        Creates a attribute-value model that uses the probabilities appearing in unimorph data.
        """
        values = []
        probs = []
        for cache_entry in cache_entries:
            values.append(cache_entry.get_value())
            probs.append(cache_entry.get_sampling_prob(as_torch=False))

        if not isclose(sum(probs), 1.0):
            raise Exception("Attribute-value model probabilities do not add up to one.")

        return cls(list(zip(values, probs)), device=device)

    def get_num_possibilities(self) -> int:
        return len(self._probs)

    def get_probs(self) -> List[float]:
        return list(self._probs)

    def sample(self, num_samples) -> torch.Tensor:
        samples = self._distribution.sample(sample_shape=torch.Size([num_samples]))
        assert samples.shape == (num_samples,)
        return samples

    def log_prob(self, sample) -> torch.Tensor:
        probs = self._distribution.log_prob(sample)
        assert probs.shape == (sample.shape[0],)
        return probs

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()

    def get_value_ids(self, values: List[str]) -> torch.Tensor:
        value_indices = [self._values_dict[v] for v in values]
        return torch.tensor(value_indices).to(self._device)

    def get_values(self) -> List[str]:
        return list(self._values)

    def get_values_from_ids(self, values_idx: List[int]) -> List[str]:
        return [self._values[i] for i in values_idx]


class EmbeddingModel:
    def __init__(self, embedding_distributions: List[EmbeddingDistribution],
                 device: Optional[PyTorchDevice] = "cpu"):
        self._device = device
        self._embedding_size = embedding_distributions[0].get_dimensionality()
        self._embedding_dists = embedding_distributions

    def get_dimensionality(self) -> int:
        return self._embedding_size

    def sample(self, attribute_values):
        num_samples = attribute_values.shape[0]

        # Sample from all embedding distributions
        dist_results = [x.sample(num_samples=num_samples) for x in self._embedding_dists]

        dist_results_stacked = torch.stack(dist_results, dim=2)
        assert dist_results_stacked.shape == (num_samples, self._embedding_size, len(self._embedding_dists))

        # Select correct sampled embedding depending on which value it was sampled from
        mask = torch.arange(0, len(self._embedding_dists)).reshape(1, -1).expand(
            num_samples, -1).to(self._device) == attribute_values.unsqueeze(1).expand(-1, len(self._embedding_dists))
        mask = mask.unsqueeze(1)

        correct_samples = torch.sum(dist_results_stacked * mask, dim=2)
        assert correct_samples.shape == (num_samples, self._embedding_size)

        return correct_samples

    def log_prob(self, samples):
        """
        Returns probabilities of samples according to ALL embedding distributions,
        aka. forall v in calV   p(h_C|v,a)  # noqa
        """
        num_samples = samples.shape[0]

        # Compute sample probs according to each distribution
        log_probs = [x.log_prob(samples) for x in self._embedding_dists]
        log_probs = torch.stack(log_probs, dim=1)
        assert log_probs.shape == (num_samples, len(self._embedding_dists))

        return log_probs

    def log_prob_conditional(self, samples, attribute_values):
        """
        Returns probabilities of samples conditioned on the values in attribute_values, aka. p(h_C|v,a)
        """
        num_samples = samples.shape[0]
        log_probs = self.log_prob(samples)

        # Select correct sample log prob according to which value it was sampled from
        mask = torch.arange(0, len(self._embedding_dists)).reshape(1, -1).expand(
            num_samples, -1).to(self._device) == attribute_values.unsqueeze(1).expand(-1, len(self._embedding_dists))
        log_probs = torch.sum(mask * log_probs, dim=1)
        assert log_probs.shape == (num_samples,)

        return log_probs


class ProbingModel:
    def __init__(self, embedding_model: EmbeddingModel, value_model: ValueModel,
                 device: PyTorchDevice = "cpu"):
        self._embedding_model = embedding_model
        self._value_model = value_model
        self._device = device

    def get_device(self) -> PyTorchDevice:
        return self._device

    def get_pred_true_arrays(self, attribute: str, select_dimensions: Iterable[int],
                             eval_dataset: Reader) -> Tuple[torch.Tensor, torch.Tensor, List[Word]]:
        if attribute not in eval_dataset.get_valid_attributes():
            raise Exception("The attribute '{}' does not exist in the evaluation dataset.".format(attribute))

        select_dimensions = list(select_dimensions)

        # Select words that have the attribute specified attribute
        cache_key = f"words_with_{attribute}"
        cache_key_filter = lambda x: x.has_attribute(attribute)  # noqa

        words = eval_dataset.get_words_with_filter_from_cache(
            cache_key, cache_key_filter)

        values_tensor = eval_dataset.get_values_with_filter_from_cache(
            attribute, cache_key, self._value_model, cache_key_filter)
        embeddings_tensor = eval_dataset.get_embeddings_with_filter_from_cache(
            cache_key, cache_key_filter)[:, select_dimensions].to(self._device)

        conditional_probs = self.log_prob_all_values(embeddings_tensor)
        predicted_values = conditional_probs.argmax(dim=1)

        return predicted_values, values_tensor, words

    def get_log_likelihood(self, attribute: str, select_dimensions: Iterable[int],
                           eval_dataset: Reader) -> torch.Tensor:
        """
        Returns the average log likelihood.
        """
        if attribute not in eval_dataset.get_valid_attributes():
            raise Exception("The attribute '{}' does not exist in the evaluation dataset.".format(attribute))

        select_dimensions = list(select_dimensions)

        # Select words that have the attribute specified attribute
        cache_key = f"words_with_{attribute}"
        cache_key_filter = lambda x: x.has_attribute(attribute)  # noqa

        values_tensor = eval_dataset.get_values_with_filter_from_cache(
            attribute, cache_key, self._value_model, cache_key_filter)
        embeddings_tensor = eval_dataset.get_embeddings_with_filter_from_cache(
            cache_key, cache_key_filter)[:, select_dimensions].to(self._device)

        # Compute probabilities
        log_prob = self.log_prob_conditional(embeddings_tensor, values_tensor)
        log_prob_normalizer = self.log_prob(embeddings_tensor)

        log_prob_avg = torch.mean(log_prob - log_prob_normalizer)
        return log_prob_avg

    @staticmethod
    def get_accuracy(predicted_values: torch.Tensor, values_tensor: torch.Tensor) -> torch.Tensor:
        num_items = predicted_values.shape[0]
        return (predicted_values == values_tensor).sum().float() / num_items

    def get_value_model(self) -> ValueModel:
        return self._value_model

    def sample(self, num_samples) -> torch.Tensor:
        # Sample values from categorical
        samples = self._value_model.sample(num_samples)

        # Using sampled categorical values, sample from correct embedding distributions
        emb_samples = self._embedding_model.sample(samples)

        return emb_samples

    def log_prob_all_values(self, samples) -> torch.Tensor:
        """
        Returns p(h_C,v|a) for all v
        """
        num_samples = samples.shape[0]

        # Get log probs for all possibilies ways to sample the values
        val_log_probs = [self._value_model.log_prob((torch.ones(num_samples).long() * x).to(self._device))
                         for x in range(self._value_model.get_num_possibilities())]
        val_log_prob = torch.stack(val_log_probs, dim=1)

        # Get log probs for all possible values embeddings may have been sampled from
        emb_log_prob = self._embedding_model.log_prob(samples)
        assert emb_log_prob.shape == val_log_prob.shape

        # Numerically stable computation
        log_prob_sum = emb_log_prob + val_log_prob

        return log_prob_sum

    def log_prob(self, samples):
        """
        Returns p(h_C|a)
        """
        num_samples = samples.shape[0]
        log_prob_sum = self.log_prob_all_values(samples)
        log_prob = log_prob_sum.logsumexp(dim=1)

        assert log_prob.shape == (num_samples,)
        return log_prob

    def log_prob_conditional(self, samples, attribute_values):
        """
        Return p(h_C, v|a) = p(h_C|v, a) p(v|a)
        """
        num_samples = samples.shape[0]

        emb_log_prob = self._embedding_model.log_prob_conditional(samples, attribute_values)
        val_log_prob = self._value_model.log_prob(attribute_values)
        log_prob = val_log_prob + emb_log_prob
        assert log_prob.shape == (num_samples,)

        return log_prob
