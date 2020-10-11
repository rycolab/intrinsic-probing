# Intrinsic Probing through Dimension Selection

This is the repository for "Intrinsic Probing through Dimension Selection".

## Setup

These instructions assume that conda is already installed on your system.

1. Clone this repository. *NOTE: We recommend keeping the default folder name when cloning.*
2. First run `conda env create -f environment.yml`.
3. Activate the environment with `conda activate intrinsic-probing`.
4. Install [pyTorch](https://pytorch.org/get-started/locally/).
5. Install [torch-scatter](https://github.com/rusty1s/pytorch_scatter).
6. (Optional) Setup wandb, if you want live logging of your runs.
7. You will also need to install fastText to your environment as described [here](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module).
8. Setup the config file `cp config.default.py config.py`.

### Generate data

You will also need to generate the data. Here we provide instructions on how to obtain the data to replicate our entire study.

1. First run `mkdir unimorph && cd unimorph && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml`
2. Download [UD 2.1 treebanks](http://hdl.handle.net/11234/1-2515) and put them in `data/ud/ud-treebanks-v2.1`
3. Download all fastText embedding files by running `cd scripts; ./download_fasttext_vectors.sh; cd ..`. **WARNING: This may take a while & require a lot of bandwidth.**
4. Clone the modified [UD converter](git@github.com:ltorroba/ud-compatibility.git) to this repository's parent folder and then convert the treebank annotations to the UniMorph schema with `cd scripts; ./ud_to_um.sh; cd ..`. *NOTE: This step will fail if the repositories were cloned into folders different than the default. If you changed the folder name, you can update the top lines in the shell file to reflect that.*
5. Run `./scripts/preprocess_bert.sh` to preprocess all the relevant treebanks using BERT. This may take a while.
6. Run `./scripts/preprocess_fasttext.sh` to preprocess all the relevant treebanks using FastText. This may take a while.
7. (Only on a headless server) Orca needs X11 to run, or else it cannot generate graphs. An easier alternative is to run `sudo apt-get install xvfb` and then open a `python` interpreter and run:
    ```
    >>> import plotly.io as pio
    >>> pio.orca.config.use_xvfb = True
    >>> pio.orca.config.save()
    ```

## Running

All the experiments are run using `run_ud_treebanks.py`.
For a list of options you can use, run `python run_ud_treebanks.py -h`.

For example, to replicate our MAP experiments for Portuguese fastText, you would run `python run_ud_treebanks.py por fasttext --max-iter 50 --trainer map`.

You can also run `./scripts/run_ud_all_experiments.sh` to reproduce experimental results.
