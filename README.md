# Weisfeiler and Leman go Hyperbolic: Learning Distance Preserving Node Representations (WLHN)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the original implementation of the paper titled [Weisfeiler and Leman go Hyperbolic: Learning Distance Preserving Node Representations](https://arxiv.org/abs/2211.02501), accepted at AISTATS2023.

## Requirements
Libraries: pytorch, pytorch_geometric


## Usage

To train our model WLHN, use the above command:
```
python tu_dataset.py 
```
Arguments:
```
--dataset "Dataset name"
--lr "Initial Learning rate"
--dropout "Dropout rate"
--batch-size "Input batch size for training"
--epochs "Number of epochs to train"
--hidden-dim "Size of hidden layer"
--tau "Tau value for Sarkar's construction"
--depth "Depth of WL tree"
--classifier " Classifier (hyperbolic_mlr, logmap)
--hyperbolic-optimizer "Whether to use hyperbolic optimizer"
```
## Contribution
- Giannis Nikolentzos
- Michail Chatzianastasis  
- Michalis Vazirgiannis 





 

