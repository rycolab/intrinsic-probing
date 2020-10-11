from typing import Any, Dict, List, Set, Optional
from tqdm import tqdm
import torch
from uncertainties import ufloat

from models.base import ProbingModel
from models.gaussian_model import GaussianEmbeddingModel
from estimators.fixed_samples import FixedSamplesEstimator
from utils.graph_writer import GraphWriter

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class Runner:
    def __init__(self, config: Dict[str, Any]):
        self.reader = config["reader"]
        self.reader_valid = config["reader_valid"]
        self.reader_test = config["reader_test"]
        self.device = config["device"]
        self.cache = config["cache"]
        self.cache_valid = config["cache_valid"]
        self.cache_test = config["cache_test"]
        self.value_model = config["value_model"]
        self.value_model_valid = config["value_model_valid"]
        self.value_model_test = config["value_model_test"]
        self.attribute = config["attribute"]
        self.selection_criterion = config["selection_criterion"]
        self.wandb_run = config["wandb_run"] if "wandb_run" in config else None

        # Create caches for train & test set
        attribute_values = self.cache.get_all_attribute_values(self.attribute)
        cache_entries = [self.cache.get_cache_entry(self.attribute, v) for v in attribute_values]
        self.cache_entries = cache_entries

    def main_loop(self, max_iter: int = 10) -> List[Dict[str, Any]]:
        total_dims = self.reader.get_dimensionality()
        selected_dims: Set[int] = set()
        selected_results: List[Dict[str, Any]] = []

        for iter_idx in range(max_iter):
            iteration_results = []
            unselected_dims = set(list(range(total_dims))) - selected_dims

            # Compute all possible next dims
            for candidate_dim in tqdm(unselected_dims):
                candidate_dim_pool = selected_dims | {candidate_dim}

                # Create embedding model from selected dims
                embedding_model = GaussianEmbeddingModel.from_cache_entries(
                    self.cache_entries, select_dimensions=candidate_dim_pool, device=self.device)

                # Compute metrics using validation data
                gaussian_model = ProbingModel(embedding_model, self.value_model, device=self.device)
                estimator = FixedSamplesEstimator(gaussian_model, self.reader_valid, self.attribute,
                                                  select_dimensions=candidate_dim_pool,
                                                  value_model=self.value_model_valid)
                entropy = self.value_model_valid.entropy().item() / torch.tensor(2.0).log().item()
                conditional_entropy = estimator.estimate_conditional_entropy() / torch.tensor(2.0).log().item()
                mutual_information = entropy - conditional_entropy

                y_pred, y_true, words = gaussian_model.get_pred_true_arrays(
                    self.attribute, candidate_dim_pool, self.reader_valid)
                accuracy = gaussian_model.get_accuracy(y_pred, y_true).cpu().item()

                log_likelihood = gaussian_model.get_log_likelihood(
                    self.attribute, candidate_dim_pool, self.reader_valid)

                iteration_results.append({
                    "candidate_dim": candidate_dim,
                    "candidate_dim_pool": list(candidate_dim_pool),
                    "mi": mutual_information,
                    "accuracy": accuracy,
                    "conditional_entropy": conditional_entropy,
                    "entropy": entropy,
                    "log_likelihood": log_likelihood,
                })

            # Select greedily
            if self.selection_criterion == "mi":
                best_dim = max(iteration_results, key=lambda x: x["mi"])
            elif self.selection_criterion == "accuracy":
                best_dim = max(iteration_results, key=lambda x: x["accuracy"])
            elif self.selection_criterion == "log_likelihood":
                best_dim = max(iteration_results, key=lambda x: x["log_likelihood"])

            # Add selected dimension to dim pool
            selected_dims.add(best_dim["candidate_dim"])

            best_dim_test_metrics = {
                "candidate_dim": best_dim["candidate_dim"],
                "candidate_dim_pool": best_dim["candidate_dim_pool"],
            }

            best_dim_test_metrics.update(self.compute_test_metrics(best_dim["candidate_dim_pool"]))

            # Lower-bound metrics
            if len(selected_results) > 1:
                mi_max = max(selected_results[-1]["mi_max"], best_dim_test_metrics["mi"])
                model_accuracy_max = max(selected_results[-1]["model_accuracy_max"], best_dim_test_metrics["model_accuracy"])
            else:
                mi_max = max(best_dim_test_metrics["mi"], ufloat(0.0, 0.0))
                model_accuracy_max = best_dim_test_metrics["model_accuracy"]

            best_dim_test_metrics.update(mi_max=mi_max)
            best_dim_test_metrics.update(model_accuracy_max=model_accuracy_max)

            selected_results.append(best_dim_test_metrics)
            print("Selected '{}'".format(best_dim_test_metrics["candidate_dim"]))
            print("\tI(H_{}; V_a): {}".format(best_dim_test_metrics["candidate_dim_pool"],
                                              best_dim_test_metrics["mi"].nominal_value))
            print("\tAccuracy: {}".format(best_dim_test_metrics["model_accuracy"]))
            print("\tConfusion Matrix: ")
            print(best_dim_test_metrics["confusion_matrix"])
            print()

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "mi": best_dim_test_metrics["mi"].nominal_value,
                    "mi_max": best_dim_test_metrics["mi_max"].nominal_value,
                    "conditional_entropy": best_dim_test_metrics["conditional_entropy"].nominal_value,
                    "entropy": best_dim_test_metrics["entropy"],
                    "mi_normalized": best_dim_test_metrics["mi"].nominal_value / best_dim_test_metrics["entropy"],  # noqa
                    "mi_max_normalized": best_dim_test_metrics["mi_max"].nominal_value / best_dim_test_metrics["entropy"],  # noqa
                    "conditional_entropy_normalized": best_dim_test_metrics["conditional_entropy"].nominal_value / best_dim_test_metrics["entropy"],  # noqa
                    "model_accuracy": best_dim_test_metrics["model_accuracy"],
                    "model_accuracy_max": best_dim_test_metrics["model_accuracy_max"],
                    "baseline_accuracy": best_dim_test_metrics["baseline_accuracy"],
                    "micro_precision": best_dim_test_metrics["micro_precision"],
                    "micro_recall": best_dim_test_metrics["micro_recall"],
                    "micro_f1": best_dim_test_metrics["micro_f1"],
                    "macro_precision": best_dim_test_metrics["macro_precision"],
                    "macro_recall": best_dim_test_metrics["macro_recall"],
                    "macro_f1": best_dim_test_metrics["macro_f1"],
                })

        # Get maximum accuracy
        full_vector_test_metrics = self.compute_test_metrics(list(range(total_dims)))
        print("Full Vector Accuracy: {}".format(full_vector_test_metrics["model_accuracy"]))

        # Add summary metrics (every 5 steps)
        if self.wandb_run is not None:
            # Add entropy
            self.wandb_run.summary["entropy"] = selected_results[-1]["entropy"]

            # Add after 2 selected
            if len(selected_results) >= 2:
                idx = 1
                self.wandb_run.summary[f"mi_{idx + 1}"] = selected_results[idx]["mi"].nominal_value
                self.wandb_run.summary[f"mi_max_{idx + 1}"] = selected_results[idx]["mi_max"].nominal_value
                self.wandb_run.summary[f"mi_normalized_{idx + 1}"] = \
                    selected_results[idx]["mi"].nominal_value / selected_results[-1]["entropy"]
                self.wandb_run.summary[f"mi_max_normalized_{idx + 1}"] = \
                    selected_results[idx]["mi_max"].nominal_value / selected_results[-1]["entropy"]
                self.wandb_run.summary[f"model_accuracy_{idx + 1}"] = selected_results[idx]["model_accuracy"]
                self.wandb_run.summary[f"model_accuracy_max_{idx + 1}"] = selected_results[idx]["model_accuracy_max"]

            # Add every 5 selected
            for pos in range(0, len(selected_results) + 1, 5)[1:]:
                idx = pos - 1
                self.wandb_run.summary[f"mi_{idx + 1}"] = selected_results[idx]["mi"].nominal_value
                self.wandb_run.summary[f"mi_max_{idx + 1}"] = selected_results[idx]["mi_max"].nominal_value
                self.wandb_run.summary[f"mi_normalized_{idx + 1}"] = \
                    selected_results[idx]["mi"].nominal_value / selected_results[-1]["entropy"]
                self.wandb_run.summary[f"mi_max_normalized_{idx + 1}"] = \
                    selected_results[idx]["mi_max"].nominal_value / selected_results[-1]["entropy"]
                self.wandb_run.summary[f"model_accuracy_{idx + 1}"] = selected_results[idx]["model_accuracy"]
                self.wandb_run.summary[f"model_accuracy_max_{idx + 1}"] = selected_results[idx]["model_accuracy_max"]

            # Report full vector accuracy as summary in wandb
            self.wandb_run.summary["model_accuracy_full"] = full_vector_test_metrics["model_accuracy"]

        return selected_results

    def compute_test_metrics(self, candidate_dim_pool: List[int]) -> Dict[str, Any]:
        # Create embedding model from selected dims
        embedding_model = GaussianEmbeddingModel.from_cache_entries(
            self.cache_entries, select_dimensions=candidate_dim_pool, device=self.device)

        # Compute mutual information
        gaussian_model = ProbingModel(embedding_model, self.value_model, device=self.device)
        estimator = FixedSamplesEstimator(gaussian_model, self.reader_test, self.attribute,
                                          select_dimensions=candidate_dim_pool, value_model=self.value_model_test)
        entropy = self.value_model_test.entropy().item() / torch.tensor(2.0).log().item()
        conditional_entropy = estimator.estimate_conditional_entropy() / torch.tensor(2.0).log().item()
        mutual_information = entropy - conditional_entropy

        y_pred, y_true, words = gaussian_model.get_pred_true_arrays(
            self.attribute, candidate_dim_pool, self.reader_test)
        y_true_label = y_true.cpu().tolist()
        y_pred_label = y_pred.cpu().tolist()

        # Class-wise metrics
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            y_true_label, y_pred_label, average=None)
        # Support: the number of items for each class in y_true
        class_support = class_support.tolist()

        # Macro metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label, average="macro")

        # Micro metrics
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label, average="micro")

        # Log-likelihood
        log_likelihood = gaussian_model.get_log_likelihood(
            self.attribute, candidate_dim_pool, self.reader_test)

        return {
            "mi": mutual_information,
            "conditional_entropy": conditional_entropy,
            "entropy": entropy,
            "model_accuracy": gaussian_model.get_accuracy(y_pred, y_true).cpu().item(),
            # Most common item baseline accuracy
            "baseline_accuracy": max(class_support) / sum(class_support),
            "test_predictions": [{
                "word": w.get_word(),
                "pred": y_pred_label[idx],
                "true": y_true_label[idx]
            } for idx, w in enumerate(words)],
            "confusion_matrix": confusion_matrix(y_true_label, y_pred_label).tolist(),
            "class_precision": class_precision.tolist(),
            "class_recall": class_recall.tolist(),
            "class_f1": class_f1.tolist(),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "log_likelihood": log_likelihood.cpu().item(),
        }

    def draw_graphs(self, selected_results) -> Dict[str, Any]:
        entropy = selected_results[-1]["entropy"]
        gw = GraphWriter(selected_results)
        mi_fig = gw.plot_mi(entropy, practical_maximum=selected_results[-1]["mi_max"].nominal_value)
        normalized_mi_fig = gw.plot_normalized_mi(
            entropy, practical_maximum=selected_results[-1]["mi_max"].nominal_value / entropy)
        accuracy_fig = gw.plot_accuracy()
        return {
            "mi": mi_fig,
            "normalized_mi": normalized_mi_fig,
            "accuracy": accuracy_fig
        }

    def plot_dims(
            self, dim_1: int, dim_2: int, test_data: bool = False,
            log_prob_dim_pool: Optional[List[int]] = None,
            show_legend: bool = False):
        data: GraphWriter.DimensionScatterGraphDataType = []
        reader = self.reader_test if test_data else self.reader

        for cache_entry in self.cache_entries:
            value = cache_entry.get_value()
            points = []
            words = [
                w for w in reader.get_words()
                if w.has_attribute(self.attribute) and w.get_attribute(self.attribute) == value
            ]

            for w in words:
                # Add base info
                info: Dict[str, Any] = {
                    "word": w.get_word(), "coordinate": (w.get_embedding()[dim_1], w.get_embedding()[dim_2])
                }

                points.append(info)

            # Add datapoint log probs if a log prob dim pool has been passed
            if log_prob_dim_pool is not None:
                embedding_model = GaussianEmbeddingModel.from_cache_entries(
                    self.cache_entries, select_dimensions=log_prob_dim_pool, device=self.device)
                gaussian_model = ProbingModel(embedding_model, self.value_model, device=self.device)

                attribute_values = (gaussian_model.get_value_model().get_value_ids([value])).expand(len(words))
                samples = torch.stack([torch.tensor(w.get_embedding()[log_prob_dim_pool]) for w in words], dim=0).to(self.device)

                log_probs_value = gaussian_model.log_prob_conditional(samples, attribute_values)
                log_probs = log_probs_value - gaussian_model.log_prob(samples)
                log_probs = log_probs.cpu().tolist()

                for info, log_prob in zip(points, log_probs):
                    info["log_prob"] = log_prob

            data.append((cache_entry, points))

        fig = GraphWriter.plot_dimension_scatter_graph(data, dim_1, dim_2, device=self.device, show_legend=show_legend)

        return fig
