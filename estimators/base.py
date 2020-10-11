from typing import Optional
import torch
from uncertainties import ufloat

from models.base import ProbingModel, ValueModel


class Estimator:
    """
    Base class for an MI estimator.
    """
    def __init__(self, probing_model: ProbingModel, value_model: Optional[ValueModel] = None):
        self._probing_model = probing_model
        self._value_model = value_model or self._probing_model.get_value_model()

    def estimate_integral(self, value_name: str) -> ufloat:
        """
        Estimates the integral we need to compute
        """
        raise NotImplementedError

    def estimate_conditional_entropy(self) -> ufloat:
        conditional_prob = ufloat(0.0, 0.0)
        with torch.no_grad():
            for v_name, val_prob in zip(self._value_model.get_values(), self._value_model.get_probs()):
                integral_estimate = self.estimate_integral(v_name)
                local_conditional_prob = -val_prob * integral_estimate
                conditional_prob += local_conditional_prob

        return conditional_prob
