conda create -n HuggingfaceActiveLearning \
python=3.11 \
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
conda activate HuggingfaceActiveLearning
pip install evaluate
