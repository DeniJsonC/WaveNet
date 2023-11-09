# Installation

This repository is built in PyTorch 1.11 and tested on Ubuntu 18.04 environment (Python3.8, CUDA11.7).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/DeniJsonC/WaveNet.git
cd WaveNet
```

2. Make conda environment
```
conda create -n pytorch111 python=3.8
conda activate pytorch111
```

3. Install dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.7 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext
```
