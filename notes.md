## Installation
```bash
conda create -y -n tfm python=3.11
source activate tfm
pip install -e .
pip install tabicl
```

## Training
```bash
python pretrain_classification.py --epochs 10 --steps 25 --batchsize 50 --priordump data/50x3_3_100k_classification.h5
```