# MDALTH: Modular Deep Active Learning on Torch and Huggingface

MDALTH is a modular library of active learning query algorithms and stopping methods geared towards torch and huggingface users.

## Setup

Install git and a package manager, e.g., conda on a Linux machine with a GPU (or a cluster with many GPUs, if you have one lying around).

You do you, but we like conda. To create a virtual environment and install the requirements
```
conda create -n MDALTH \
python=3.11 \
pytorch-cuda=11.8 \
pytorch \
torchvision \
torchaudio \
torchtext \
transformers \
datasets \
tokenizers \
accelerate \
scipy \
scikit-learn \
pandas \
matplotlib \
-c pytorch -c nvidia -c conda-forge
conda activate MDALTH
pip install evaluate
```

Obviously, you need to `conda activate` it each time before use. 

We are working on improving the flexibility of our dependencies, but for now, see environment.yml for comprehenisve details about dependency requirements.

Next, clone the repository
```
git clone git@github.com:lkurlandski/MDALT.git
```

To use mdalth components, you have two choices. You can either work within the MDALTH directory as is done in our examples, or you can pip install MDALTH directly into your environment. To do the latter,

```
pip install -e .
```

## Similar Libraries

Several Python libraries for active learning have already been proposed, however, have significant disadvantages when compared to MDAL. Notably, the community still lacks an open-source library of AL stopping methods, which are a crucial aspect of the AL pipeline.

### [ModAL](https://github.com/modAL-python/modAL)

ModAL wraps scikit-learn classifiers, and as such, is ill-suited for deep learning.

### [ALiPy](https://github.com/NUAA-AL/ALiPy)

ALiPy is similar to ModAL and has the exact same shortcommings.

### [deep-active-learning](https://github.com/ej0cl6/deep-active-learning)

While designed to support deep learning through Pytorch, this repo is poorly engineered, documented, and maintained.

### [badge](https://github.com/JordanAsh/badge)

BADGE is forked from deep-active-learning. While it implements some newer querier algorithms, the repository is nowhere near capable of providing a modular toolkit or AL practitioners.

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
