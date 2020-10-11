import json
from os import path
from argparse import ArgumentParser

import models.gaussian_model
from readers.ud_treebank_reader import UDTreebankReader
from utils.cache import AttributeValueGaussianCache
from utils.json import ResultsEncoder
from runner.runner import Runner

from trainer.mle import MLETrainer
from trainer.map import MAPTrainer

# Workaround to bug when generating big images (sometimes occurs)
import plotly.io._orca
import retrying
unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
plotly.io._orca.request_image_with_retrying = wrapped


# Setup arguments
parser = ArgumentParser()
parser.add_argument("language")
parser.add_argument("embedding", choices=["fasttext", "bert"])
parser.add_argument("--attribute", type=str)
parser.add_argument("--trainer", type=str, choices=["mle", "map"], default="map")
parser.add_argument("--max-iter", type=int, default=10)
parser.add_argument("--log-wandb", default=False, action="store_true")
parser.add_argument("--show-charts", default=False, action="store_true")
parser.add_argument("--use-gpu", default=False, action="store_true")
parser.add_argument("--selection-criterion", choices=["accuracy", "mi", "log_likelihood"], default="log_likelihood")
parser.add_argument("--tag", type=str)
parser.add_argument(
    "--diagonalize", default=False, action="store_true", help="If set diagonalizes covariances, aka. uses a naive model \
        that assumes conditional independence of embedding dimensions given the class.")
args = parser.parse_args()

device = "cpu"
if args.use_gpu:
    print("Using gpu:0")
    device = "cuda:0"

# Create Reader for English UD Treebank
treebank = UDTreebankReader.get_treebank_file(args.language, embedding=args.embedding)
treebank_valid = UDTreebankReader.get_treebank_file(args.language, embedding=args.embedding, valid_file=True)
treebank_test = UDTreebankReader.get_treebank_file(args.language, embedding=args.embedding, test_file=True)

print(f"Treebank: {treebank}")
print(f"Valid Treebank: {treebank_valid}")
print(f"Test Treebank: {treebank_test}")

words = UDTreebankReader.read([treebank])
words_valid = UDTreebankReader.read([treebank_valid])
words_test = UDTreebankReader.read([treebank_test])

counters = [
    UDTreebankReader.get_attribute_value_counter(words),
    UDTreebankReader.get_attribute_value_counter(words_valid),
    UDTreebankReader.get_attribute_value_counter(words_test)
]
attr_vals_dict = UDTreebankReader.get_attributes_to_values_dict_from_counters(counters, min_count=100)

reader = UDTreebankReader(words, attr_vals_dict)
reader_valid = UDTreebankReader(words_valid, attr_vals_dict)
reader_test = UDTreebankReader(words_test, attr_vals_dict)

# Build caches
print("Building caches...")
if args.trainer == "mle":
    trainer = MLETrainer()
elif args.trainer == "map":
    # trainer = MAPTrainer.from_dimension(device=device, dimension=reader.get_dimensionality())
    trainer = MAPTrainer.from_data(device=device)

cache_attr_vals_dict = attr_vals_dict
if args.attribute is not None:
    cache_attr_vals_dict = {k: v for k, v in attr_vals_dict.items() if k == args.attribute}

cache = AttributeValueGaussianCache(
    reader.get_words(), trainer=trainer, attribute_values_dict=cache_attr_vals_dict, diagonal_only=args.diagonalize)
cache_valid = AttributeValueGaussianCache(
    reader_valid.get_words(), trainer=trainer, attribute_values_dict=cache_attr_vals_dict, diagonal_only=args.diagonalize)
cache_test = AttributeValueGaussianCache(
    reader_test.get_words(), trainer=trainer, attribute_values_dict=cache_attr_vals_dict, diagonal_only=args.diagonalize)

if args.attribute is not None:
    attributes_queue = [args.attribute]
else:
    attributes_queue = list(attr_vals_dict.keys())

    ignore_list = ["Part of Speech"]
    attributes_queue = [x for x in attributes_queue if x not in ignore_list]

print(f"Attributes queue: {attributes_queue}")

for attribute in attributes_queue:
    # Safety checks--do not want to initialize a run that we can't actually do
    if not cache.has_attribute(attribute):
        print(f"Attribute '{attribute}' does not exist in this dataset/language combination.")
        exit()

    if len(cache.get_all_attribute_values(attribute)) < 2:
        print(f"Attribute '{attribute}' has less that 2 values in this dataset/language combination.")
        exit()

    if args.log_wandb:
        import wandb
        tags = [args.language, args.embedding, attribute]
        args.attribute = attribute
        if args.tag is not None:
            tags.append(args.tag)

        run = wandb.init(project="interp-bert", tags=tags, config=args, reinit=True)
        run.name = f"{attribute} ({args.embedding}-{args.language}) ("
        if args.diagonalize:
            run.name += f"{args.selection_criterion}, diag)"
        else:
            run.name += f"{args.selection_criterion})"

        run.name += f" [{wandb.run.id}]"
        run.save()

    print("Computing MI for '{}'. Possible values: {}".format(
        attribute, cache.get_all_attribute_values(attribute)))

    # Create value model
    attribute_values = cache.get_all_attribute_values(attribute)
    value_model = models.base.ValueModel.from_cache_entries(
        [cache.get_cache_entry(attribute, v) for v in attribute_values], device=device
    )

    value_model_valid = models.base.ValueModel.from_cache_entries(
        [cache_valid.get_cache_entry(attribute, v) for v in attribute_values], device=device
    )

    value_model_test = models.base.ValueModel.from_cache_entries(
        [cache_test.get_cache_entry(attribute, v) for v in attribute_values], device=device
    )

    runner_config = {
        "reader": reader,
        "reader_valid": reader_valid,
        "reader_test": reader_test,
        "device": device,
        "cache": cache,
        "cache_valid": cache_valid,
        "cache_test": cache_test,
        "value_model": value_model,
        "value_model_valid": value_model_valid,
        "value_model_test": value_model_test,
        "attribute": attribute,
        "selection_criterion": args.selection_criterion,
    }

    if args.log_wandb:
        runner_config["wandb_run"] = run

    total_dims = reader.get_dimensionality()
    runner = Runner(runner_config)
    selected_results = runner.main_loop(max_iter=args.max_iter)

    # Draw graphs
    graphs = runner.draw_graphs(selected_results)
    mi_fig = graphs["mi"]
    normalized_mi_fig = graphs["normalized_mi"]
    accuracy_fig = graphs["accuracy"]
    scatter_fig = runner.plot_dims(
        selected_results[0]["candidate_dim"], selected_results[1]["candidate_dim"], test_data=True,
        log_prob_dim_pool=list(selected_results[-1]["candidate_dim_pool"])
    )

    # You can uncomment these lines to output scatter plots for any pair of dimensions you need.
    #
    # scatter_fig = runner.plot_dims(477, 179, test_data=True)
    # scatter_fig.write_image(f"images/scatter_{args.embedding}_{args.language}_{args.attribute}.pdf")
    # scatter_fig.write_html(f"images/scatter_{args.embedding}_{args.language}_{args.attribute}.pdf")

    if args.show_charts:
        mi_fig.show()
        normalized_mi_fig.show()
        accuracy_fig.show()
        scatter_fig.show()

    # Save selected results
    if args.log_wandb:
        with open(path.join(run.dir, "results.json"), "w") as h:
            json.dump(selected_results, h, cls=ResultsEncoder)

        mi_fig.write_image(path.join(run.dir, "mi_result.pdf"))
        normalized_mi_fig.write_image(path.join(run.dir, "normalized_mi_result.pdf"))
        accuracy_fig.write_image(path.join(run.dir, "accuracy_result.pdf"))
        scatter_fig.write_image(path.join(run.dir, "scatter_result.pdf"))
        run.log({"graph_mi": mi_fig})
        run.log({"graph_normalized_mi": normalized_mi_fig})
        run.log({"graph_accuracy": accuracy_fig})
        run.log({"graph_scatter": scatter_fig})

        wandb.join()
