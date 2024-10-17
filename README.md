# SAFT

This repository contains the source code and datasets for running the experiments.

## Requirements

The code is written in Python 3.12.3. To run it, install the required packages using the following commands (a virtual environment is recommended):

```
pip install scikit-learn==1.5.0
pip install tqdm==4.66.2
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install pandas==2.2.2
pip install transformers==4.41.2
```

------

## Datasets
**Download processed data.** To reproduce the results in our paper, you need to first download the processed [datasets](https://www.dropbox.com/scl/fo/r1t3jjw6qnt2g5drjcsu4/ACCmsHjk3DF9jMpR0EEBkFA?rlkey=y6ci26ulfqt83r30k9tfu3wvi&st=xdlu39uw&dl=0). Place the `edge_data` folder in the same directory as the `SAFT` folder (if placed elsewhere, update the dataset path in the scripts).

**Raw data & data processing.** Raw data can be downloaded from [Amazon](https://nijianmo.github.io/amazon/index.html#code), [Goodreads](https://mengtingwan.github.io/data/goodreads.html) and [Google](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) directly. 

## Running
**Train**

```
bash script/$dataset/train_SAFT_{GNN/GAU}_$dataset.sh
```

**Test**

```
bash script/$dataset/test_SAFT_{GNN/GAU}_$dataset.sh
```

