# MDALTH: Modular Deep Active Learning on Torch and Huggingface

MDALTH is a modular library of active learning query algorithms and stopping methods.

## Goals

Our goal is to provide modular, framework agnostic query algorithms and stopping methods for active learning. Our secondary goal is to provide high-level wrappers to these methods for huggingface users to make active learning highly accessible.

## Status

Current status: prelease

### 0.0.0
 - AL learning loop
 - AL evaluation loop
 - basic query algorithms and accessbile wrappers for them
 - basic stopping algorithms

### 0.1.0
 - checkpointing system
 - optional run with full dataset

### 0.2.0
 - accessible stopping wrappers
   - break AL loop
   - dump data for large-scale evaluation

### 0.3.0
 - validation set control

### 0.4.0
 - query size control

### 0.5.0
 - testing framework
 - documentation

## Examples

MDALTH allows users to prototype active learning experiments quickly. In the `./example` directory, we developed a very simple project to run active learning experiments for text, image, and audio classification tasks. By adjusting the command-line-arguments, this example can run experiments with many different pretrained models and datasets. We run our experiments with the following hyperparameters:

- AL initial size: 10%
- AL batch size: 5%
- Max Training Epochs: 32
- Early Stopping Patience: 3

The rest of the hyperparameters are detailed in the `./example/main.sh` script, which configures and runs the experiments.

### text classification

- model: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- dataset: [ag_news](https://huggingface.co/datasets/ag_news)

Learning Curve Coming Soon!

### image classification (`example/image.sh`)

- model: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- dataset: [food101](https://huggingface.co/datasets/food101)

Learning Curve Coming Soon!

### audio classification (`example/audio.sh`)

- model: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- dataset: [speech_commands](https://huggingface.co/datasets/speech_commands)

Learning Curve Coming Soon!

## Setup

Install git and a package manager, e.g., conda on a Linux machine with a GPU (or a cluster with many GPUs, if you have one lying around).

I personally have had the most success using conda for core pytorch installs, then pip for other libraries because it is much faster.
```
conda create -n MDALTH python=3.11 pytorch-cuda=11.8 pytorch torchvision torchaudio torchtext -c pytorch -c nvidia
conda activate MDALTH
pip install transformers datasets tokenizers accelerate evaluate scipy scikit-learn matplotlib pandas librosa
```

Obviously, you need to `conda activate` it each time before use. If you aren't interested in doing text, image, or audio classification, you don't need to install those dependencies (e.g., torchaudio and soundfile for audio classification).

We are working on improving the flexibility of our dependencies, but for now, see environment.yml for comprehenisve details about dependency requirements.

Next, clone the repository
```
git clone git@github.com:lkurlandski/MDALTH.git
```

To use mdalth components, you have two choices. You can either work within the MDALTH directory as is done in our examples, or you can pip install MDALTH directly into your environment. To do the latter,

```
pip install -e .
```

### Problems

There may be some issues with cuda 11.8, discussed at [Issue 97041](https://github.com/pytorch/pytorch/issues/97041). If you get a warning about convolutional layers, try the solution below:

```
cd ~/anaconda/envs/MDALTH/lib
sudo ln -sfn libnvrtc.so.11.8.89 libnvrtc.so
```

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
