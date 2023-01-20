#Weisfeiler and Leman go Hyperbolic: Learning Distance Preserving Node Representations (WLHN)
This is the official implementation of the paper "Weisfeiler and Leman go Hyperbolic: Learning Distance Preserving Node Representations (WLHN)" https://arxiv.org/abs/2211.02501 , accepted at AISTATS2023

Libraries: pytorch, pytorch_geometric

To train our model WLHN, use the above command:
python tu_dataset.py 

Arguments:
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

