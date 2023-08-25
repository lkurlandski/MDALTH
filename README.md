# MDALTH: Modular Deep Active Learning for Torch and Huggingface

MDALTH is a modular library for active learning query algorithms and stopping methods. We implement the core logic of these methods using base numpy, then provide high-level wrappers for torch and huggingface users. Our goal is to make state-of-the-art AL techniques accessible to all.

## Setup

Install git and a package manager, e.g., conda on a Linux machine with a GPU (or a cluster with many GPUs, if you have one lying around).

Clone the repository
```
git clone git@github.com:lkurlandski/MDALT.git
```

You do you, but we like conda. To create a virtual environment and install the requirements
```
conda create -n MDALTH \
python \
pytorch \
torchvision \
torchaudio \
torchtext \
pytorch-cuda \
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

Obviously, you need to activate each time before use. 

If you are having trouble getting things to run, see the environment.yml file for an exact specification of software requirements. If you're still having problems, you can try installing directly from the environment.yml file for a collection of dependencies that (I promise) will work:
```
conda install -y environment,yml
```

## Release

Our number one goal at the moment is to implement as many stopping methods and query algorithms for classification as possible. Once we feel we are ready, our 1.0 version will provide a stable, simple, and well-documented API for these tools. Until then, users will have to subclass the Learner and write their own interface to communicate between the stopping/querying algorithms and the mdalth Learner.

## Similar Libraries

Several Python libraries for active learning have already been proposed, however, have significant disadvantages when compared to MDAL. Notably, the community still lacks an open-source library of AL stopping methods, which are a crucial aspect of the AL pipeline.

### ModAL

https://github.com/modAL-python/modAL

ModAL wraps scikit-learn classifiers, and as such, is ill-suited for deep learning.

### ALiPy

https://github.com/NUAA-AL/ALiPy

ALiPy is similar to ModAL and has the exact same shortcommings.

### deep-active-learning

https://github.com/ej0cl6/deep-active-learning

While designed to support deep learning through Pytorch, this repo is poorly engineered, documented, and maintained.

### badge

https://github.com/JordanAsh/badge

BADGE is forked from deep-active-learning. While it implements some newer querier algorithms, the repository is nowhere near capable of providing a modular toolkit or AL practitioners.

## Contributing

Please consider suggestions from pylint and the associated .pylintrc file. Autoformat with black --line-length=100.

## TODO
- checkpointing system for the Learner and Evaluator
- improve file storage by serializing objects as json instead of pickle
- store the Trainer's log_history for each iteration
