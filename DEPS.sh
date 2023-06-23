conda create -n HuggingfaceActiveLearning python=3.10 pytorch torchvision torchaudio torchtext pytorch-cuda=11.8 transformers datasets tokenizers accelerate scipy scikit-learn -c pytorch -c nvidia -c conda-forge
conda activate HuggingfaceActiveLearning
pip install evaluate
