# MDALTH: Modular Deep Active Learning on Torch and Huggingface

MDALTH is a modular library of active learning query algorithms and stopping methods.

## Goals

Our goal is to provide modular, framework agnostic query algorithms and stopping methods for active learning. Our secondary goal is to provide high-level wrappers to these methods for huggingface users to make active learning highly accessible.

## Status

The main branch is currently a pre-release. Our immediate goal is to develop a reasonable API for the stopping methods, tidy up a few other incomplete features, and improve documentation for the system. We hope to release a 0.0 version shortly.

## Examples

MDALTH allows users to prototype active learning experiments quickly. All experiments are run with `./example/main.py` then analyzed with the `./example/analysis.ipynb` notebook. Each experiment uses the following experimental setup:

- AL initial size: 32
- AL batch size: 32
- AL iterations: 16
- Max Training Epochs: 24
- Early Stopping Patience: 2

### text classification

- model: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- dataset: [ag_news](https://huggingface.co/datasets/ag_news)

Coming Soon!

### image classification

- model: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- dataset: [food101](https://huggingface.co/datasets/food101)

Coming Soon!

### audio classification

- model: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- dataset: [speech_commands](https://huggingface.co/datasets/speech_commands)

Coming Soon!

## Setup

Install git and a package manager, e.g., conda on a Linux machine with a GPU (or a cluster with many GPUs, if you have one lying around).

I personally have had the most success using conda for core pytorch installs, then pip for other libraries because it is much faster.
```
conda create -n MDALTH python=3.11 pytorch-cuda=11.8 pytorch torchvision torchaudio torchtext -c pytorch -c nvidia
conda activate MDALTH
pip install transformers datasets tokenizers accelerate evaluate scipy scikit-learn matplotlib pandas
```

Obviously, you need to `conda activate` it each time before use. This command may take a day to actually execute because conda is so slow (it should really be called tortoise), so you should probably run it in a screen or tmux shell.

We are working on improving the flexibility of our dependencies, but for now, see environment.yml for comprehenisve details about dependency requirements.

Next, clone the repository
```
git clone git@github.com:lkurlandski/MDALTH.git
```

To use mdalth components, you have two choices. You can either work within the MDALTH directory as is done in our examples, or you can pip install MDALTH directly into your environment. To do the latter,

```
pip install -e .
```

See the ./example directory for examples of how to use MDALTH.

## Similar Libraries

Several Python libraries for active learning have already been proposed, however, have significant disadvantages when compared to MDALTH. Notably, the community still lacks an open-source library of AL stopping methods, which are a crucial aspect of the AL pipeline.

### [ModAL](https://github.com/modAL-python/modAL)

ModAL wraps scikit-learn classifiers, and as such, is ill-suited for deep learning.

### [ALiPy](https://github.com/NUAA-AL/ALiPy)

ALiPy is similar to ModAL and has the exact same shortcommings.

### [deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

While designed to support deep learning through Pytorch, this repo is poorly engineered, documented, and maintained.

### [badge](https://github.com/JordanAsh/badge)

BADGE is forked from deep-active-learning. While it implements some newer querier algorithms, the repository is nowhere near capable of providing a modular toolkit for AL practitioners.

## Contributing

We are interested in collaborating! Message me on Github if you would like to get involved.

### Style

Please consider suggestions from pylint and the associated .pylintrc file. Autoformat with black --line-length=100.

### TODO
- checkpointing system for the Learner and Evaluator
- improve file storage by serializing objects as json instead of pickle
- store the Trainer's log_history for each iteration
- use the newest features from the numpy.typing package to type hint dtypes
- implement utilities to facilitate large-scale stopping method analysis
  - dump stopping method information
  - determine stopping point after AL terminates
