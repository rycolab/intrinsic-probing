from typing import Tuple, List
import numpy as np
from overrides import overrides
from trainer.base import Trainer
from utils.word import Word


class MLETrainer(Trainer):
    @overrides
    def compute_gaussian_model_params_for_word_list(self, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = [np.array(x.get_embedding()) for x in words]

        # Create numpy array where rows are embeddings
        embeddings_array = np.vstack(embeddings)

        # Compute mean and covariance
        mean = np.mean(embeddings_array, axis=0)
        cov = np.cov(embeddings_array, rowvar=False)

        return mean, cov
