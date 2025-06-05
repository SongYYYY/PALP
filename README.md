[![DOI](https://zenodo.org/badge/987433729.svg)](https://doi.org/10.5281/zenodo.15597907)

# PALP: A Scalable Pretraining Framework for Link Prediction with Efficient Adaptation

This repository contains the official implementation for the paper "A Scalable Pretraining Framework for Link Prediction with Efficient Adaptation" accepted at KDD 2025.

## Overview

This repository contains the implementation for reproducing the results presented in Table 2 of our paper. The code currently supports experiments on two downstream datasets:
- `cora`
- `citeseer`

## Environment Setup

### Prerequisites
- CUDA 11.8
- Python 3.8 or higher
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PALP.git
cd PALP

# Install dependencies
pip install -r requirements.txt
```

## Pretraining and Checkpoints

The pretraining of PALP was conducted on ogbn-papers100M. Due to the large scale of this graph, which takes up ~200G to store the node features and structures, we were unable to release the processed data in the repo. To reproduce the results from our paper, we provide the pretrained checkpoints directly in the `ckpt-1` and `ckpt-2` directories.

## Data Structure

The repository contains the following key directories:

### Data Directories
- `node_data/`: Contains graph structures and node features. These files were originally processed using [TSGFM](https://github.com/CurryTang/TSGFM).
- `link_data/`: Contains training data for link prediction, including:
  - Positive/negative edges for train, validation and test
  - BUDDY features for edges (processed using [subgraph-sketching](https://github.com/melifluos/subgraph-sketching))

### Model Checkpoints
- `ckpt-1/`: Contains pretrained model checkpoints for the node module
- `ckpt-2/`: Contains pretrained model checkpoints for the edge module

Both models were pretrained on ogbn-papers100M. The configuration files for these models are:
- `model_1_config.yaml`
- `model_2_config.yaml`


## Usage

To test the pretrained checkpoints with our proposed adaptation strategy, use the following command:

```bash
python test_link_merge_all.py --data_name 'cora' --train_ratio 0.4
```

### Command Line Arguments

- `--data_name`: Name of the dataset to use ('cora' or 'citeseer')
- `--train_ratio`: Training data ratio (default: 0.4)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- We thank the authors of [TSGFM](https://github.com/CurryTang/TSGFM) for their data processing pipeline
- We thank the authors of [subgraph-sketching](https://github.com/melifluos/subgraph-sketching) for their BUDDY feature implementation
- We thank the authors of [NAGphormer](https://github.com/JHL-HUST/NAGphormer) for their implementation of NAGphormer
