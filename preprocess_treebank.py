from typing import List, Tuple, Dict, Any

import torch
from torch_scatter import scatter_mean
from tqdm import tqdm
from conllu import parse_incr, TokenList
import os
from os import path
import yaml
from transformers import BertTokenizer, BertModel
import pickle
from argparse import ArgumentParser
from embedders.embedders import FastTextEmbedder
from utils.parser import parse_unimorph_features
import config


_DEFAULT_TREEBANKS_ROOT = path.join(config.DATA_ROOT, "ud/ud-treebanks-v2.1")

parser = ArgumentParser(description="Preprocess the UD treebanks by converting their annotations to the UniMorph \
                        schema, and then obtaining token-level embeddings.")
parser.add_argument("treebank", type=str, help="The name of the treebank to be probed. This correspond to the folder \
                    the treebank is in (e.g., 'UD_Portuguese').")
parser.add_argument("--treebanks-root", type=str, default=_DEFAULT_TREEBANKS_ROOT, help="Root folder, where the \
                    treebanks are located. This should not need to change, unless you do not use the default folder \
                    structure as described in the documentation.")
parser.add_argument("--dry-run", default=False, action="store_true", help="If enabled, will not actually compute any \
                    embeddings, but go over the dataset and do everything else.")
parser.add_argument("--bert", default=None, help="If enabled, treebanks will be preprocessed for BERT. The name of the \
                    file corresponds to the (multilingual) bert embedding to use.")
parser.add_argument("--fasttext", default=None, help="If enabled, treebanks will be preprocessed for fastText. The \
                    file name corresponds to the fastText embedding file to use.")
parser.add_argument("--use-gpu", action="store_true", default=False, help="If enabled, uses the GPU to speed up \
                    preprocessing.")
parser.add_argument("--skip-existing", action="store_true", default=False, help="If enabled, will skip any embeddings \
                    that have already been preprocessed.")
args = parser.parse_args()

if not (args.bert or args.fasttext) or (args.bert and args.fasttext):
    raise Exception("Must do either FastText or BERT, but not both")

treebank_path = os.path.join(args.treebanks_root, args.treebank)
limit_number = None
bert_model = args.bert
fasttext_model = path.join(config.EMBEDDINGS_ROOT, "fasttext", args.fasttext) if args.fasttext else None
skip_existing = args.skip_existing
device = 'cpu'
if args.use_gpu:
    print("Using GPU")
    device = 0


def subword_tokenize(tokenizer: BertTokenizer, tokens: List[str]) -> List[Tuple[int, str]]:
    """
    Returns: List of subword tokens, List of indices mapping each subword token to one real token.
    """
    subtokens = [tokenizer.tokenize(t) for t in tokens]

    indexed_subtokens = []
    for idx, subtoks in enumerate(subtokens):
        for subtok in subtoks:
            indexed_subtokens.append((idx, subtok))

    return indexed_subtokens


def unimorph_feature_parser(line: List[str], i: int) -> Dict[str, str]:
    if line[i] == "_":
        return {}

    return parse_unimorph_features(line[i].split(";"))


def merge_attributes(tokens: List[str], value_to_attr_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Returns a dictionary containing Unimorph attributes, and the values taken on after the merge.
    """
    # First, build a list that naively merges everything
    merged_attributes: Dict[str, List[str]] = {}
    for t in tokens:
        for attr, val in t["um_feats"].items():
            if attr not in merged_attributes:
                merged_attributes[attr] = []

            merged_attributes[attr].append(val)

    # Second, remove attributes with multiple values (even if they are the same)
    final_attributes: Dict[str, str] = {}
    for attr, vals in merged_attributes.items():
        if len(vals) == 1:
            final_attributes[attr] = vals[0]

    return final_attributes


print(treebank_path)
for f in os.listdir(treebank_path):
    if path.isfile(path.join(treebank_path, f)) and "-um-" in f and f.endswith(".conllu"):
        filename = f
        full_path = path.join(treebank_path, filename)

        # Setup debugging tracker
        total = 0
        skipped: Dict[str, int] = {}

        # Load possible UM tags
        tags_file = "unimorph/tags.yaml"
        with open(tags_file, 'r') as h:
            _UNIMORPH_ATTRIBUTE_VALUES = yaml.full_load(h)["categories"]

        _UNIMORPH_VALUES_ATTRIBUTE = {v: k for k, vs in _UNIMORPH_ATTRIBUTE_VALUES.items() for v in vs}

        # Setup UM feature parsing
        _FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

        # Parse Conll-U files with UM
        final_token_list: List[TokenList] = []
        with open(full_path, "r") as h:
            # Setup BERT tokenizer here provisionally as we need to know which sentences have over 512 subtokens
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

            for sent_id, tokenlist in enumerate(tqdm(
                    parse_incr(h, fields=_FEATS, field_parsers={"um_feats": unimorph_feature_parser}))):
                # Only process first `limit_number` if it is set
                if limit_number is not None and sent_id > limit_number:
                    break

                # Remove virtual nodes
                tokenlist = [t for t in tokenlist if not (isinstance(t["id"], tuple) and t["id"][1] == ".")]

                # Build list of ids that are contracted
                contracted_ids: List[int] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        if t["id"][1] == "-":
                            # Range
                            contracted_ids.extend(list(range(t["id"][0], t["id"][2] + 1)))

                # Build dictionary of non-contracted token ids to tokens
                non_contracted_token_dict: Dict[int, Any] = {
                    t["id"]: t for t in tokenlist if not isinstance(t["id"], tuple)}

                # Build final list of (real) tokens, without any contractions
                # Contractions are assigned the attributes of the constituent words, unless there is a clash
                # with one attribute taking more than one value (e.g. POS tag is a frequent example), whereby
                # we discard it.
                final_tokens: List[Any] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        constituent_ids = list(range(t["id"][0], t["id"][2] + 1))
                        t["um_feats"] = merge_attributes(
                            [non_contracted_token_dict[x] for x in constituent_ids],
                            _UNIMORPH_VALUES_ATTRIBUTE)

                        # If this is a contraction, add it
                        final_tokens.append(t)
                    elif t["id"] not in contracted_ids:
                        # Check if this t is part of a contraction
                        final_tokens.append(t)

                final_tokens: TokenList = TokenList(final_tokens)

                # Skip if this would have more than 512 subtokens
                labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in final_tokens])
                subtoken_indices, subtokens = zip(*labelled_subwords)
                if len(subtokens) >= 512:
                    if "subtoken_count" not in skipped:
                        skipped["subtoken_count"] = 0

                    skipped["subtoken_count"] += 1
                    continue

                if "total_sents" not in skipped:
                    skipped["total_sents"] = 0

                skipped["total_sents"] += 1

                # Add this sentence to the list we are processing
                final_token_list.append(final_tokens)

        # Print logs:
        print("Skipped:")
        print(skipped)
        print()

        print(f"Total: {total}")

        if args.dry_run:
            print("Dry run finished.")
            continue

        final_results = []
        if args.bert:
            output_filename = filename.split('.')[0] + "-{}.pkl".format(bert_model)
            output_file = path.join(treebank_path, output_filename)

            if skip_existing and path.exists(output_file):
                print(f"Skipping {filename}. Reason: file already processed")
                continue

            print(f"Processing {filename}...")

            # Setup BERT
            model = BertModel.from_pretrained(bert_model).to(device)

            # Subtokenize, keeping original token indices
            results = []
            for sent_id, tokenlist in enumerate(tqdm(final_token_list)):
                labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in tokenlist])
                subtoken_indices, subtokens = zip(*labelled_subwords)
                subtoken_indices_tensor = torch.tensor(subtoken_indices).to(device)

                # We add special tokens to the sequence and remove them after getting the BERT output
                subtoken_ids = torch.tensor(
                    tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(subtokens))).to(device)

                results.append((tokenlist, subtoken_ids, subtoken_indices_tensor))

            # Prepare to compute BERT embeddings
            model.eval()

            # NOTE: No batching, right now. But could be worthwhile to implement if a speed-up is necessary.
            for token_list, subtoken_ids, subtoken_indices_tensor in tqdm(results):
                total += 1

                with torch.no_grad():
                    # shape: (batch_size, max_seq_length_in_batch + 2)
                    inputs = subtoken_ids.reshape(1, -1)

                    # shape: (batch_size, max_seq_length_in_batch)
                    indices = subtoken_indices_tensor.reshape(1, -1)

                    # shape: (batch_size, max_seq_length_in_batch + 2, embedding_size)
                    outputs = model(inputs)
                    final_output = outputs[0]

                    # shape: (batch_size, max_seq_length_in_batch, embedding_size)
                    # Here we remove the special tokens (BOS, EOS)
                    final_output = final_output[:, 1:, :][:, :-1, :]

                    # Average subtokens corresponding to the same word
                    # shape: (batch_size, max_num_tokens_in_batch, embedding_size)
                    token_embeddings = scatter_mean(final_output, indices, dim=1)

                # Convert to python objects
                embedding_list = [x.cpu().numpy() for x in token_embeddings.squeeze(0).split(1, dim=0)]

                for t, e in zip(token_list, embedding_list):
                    t["embedding"] = e

                final_results.append(token_list)
        elif args.fasttext:
            output_filename = filename.split('.')[0] + "-{}.pkl".format(fasttext_model.split("/")[-1])
            output_file = path.join(treebank_path, output_filename)

            if skip_existing and path.exists(output_file):
                print(f"Skipping {filename}. Reason: file already processed")
                continue

            print(f"Processing {filename}...")

            embedder = FastTextEmbedder(fasttext_model)

            for sent_id, token_list in enumerate(tqdm(final_token_list)):
                total += 1
                for t in token_list:
                    t["embedding"] = embedder.compute_embedding(t["form"])

                final_results.append(token_list)

        # Keep important parts
        final_results_filtered = []
        for row in final_results:
            for token in row:
                final_results_filtered.append({
                    "word": token["form"],
                    "embedding": token["embedding"],
                    "attributes": token["um_feats"],
                })

        # Save final results
        with open(output_file, "wb") as h:
            pickle.dump(final_results_filtered, h)
