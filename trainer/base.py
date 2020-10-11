from typing import List, Tuple
import numpy as np
from utils.word import Word


class Trainer:
    def compute_gaussian_model_params_for_word_list(self, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_gaussian_model_params_for_attribute_value(
            self, attribute: str, value: str, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This is used to extract the gaussian model parameters for an attribute-value pair.

        aka. mean and cov for p(h|v,a)
        """
        filtered_words = [x for x in words if x.has_attribute(attribute) and x.get_attribute(attribute) == value]

        return self.compute_gaussian_model_params_for_word_list(filtered_words)

    def compute_categorical_model_sampling_prob_for_attribute_value(
            self, attribute: str, value: str, words: List[Word]) -> float:
        """
        This is used to extract the probability of sampling a specific attribute-value pair.

        aka. p(v|a)

        In practice this will be estimated from a large corpus.
        """
        words_with_attribute = float(sum([x.get_count() for x in words if x.has_attribute(attribute)]))
        words_with_attribute_value = float(
            sum([x.get_count() for x in words
                if x.has_attribute(attribute) and x.get_attribute(attribute) == value]))

        return words_with_attribute_value / words_with_attribute
