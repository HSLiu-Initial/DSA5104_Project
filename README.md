A `PyTorch` Implementation of DSA5104 Project

# Data Processing

Data Processing includes:

1. Data Preprocessing.
2. Indexing and queries using elsticsearch.

runing by:

`python ./codes/preprocess/run_preprocess.py --dataset tmall`

# Run some modern CTR models

For feature interaction-based methods DeepFM:

`python run_deepfm_seq.py`

For user modeling methods DIN and DIEN:

`python run_dien_seq.py`

`python run_din_seq.py`

For sample-level retrieval-based methods RIM:

`python run_rim_seq.py`


