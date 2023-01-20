import argparse
import time
import json
import numpy as np
import networkx as nx
import geoopt as gt
import distutils.util

import torch
import torch.nn.functional as F
from torch import optim

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

from model import WLHN

unlabeled_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']

class Degree(object):
    def __call__(self, data):
        idx = data.edge_index[0]
        deg = degree(idx, data.num_nodes, dtype=torch.float32)
        data.x = deg.unsqueeze(1)
        return data

# Argument parser
parser = argparse.ArgumentParser(description='HGNN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--hidden-dim', type=int, default=16, help='Size of hidden layer of NN')
parser.add_argument('--tau', type=float, default=1., help="Tau value for Sarkar's algorithm")
parser.add_argument('--depth', type=int, default=1, help='Depth of WL tree')
parser.add_argument('--classifier', default='logmap', help='Classifier (hyperbolic_mlr, logmap or centroid_distance)')
#parser.add_argument('--n-centroids', type=int, default=200, help='Number of centroids in case centroid_distance classifier is employed')
parser.add_argument('--hyperbolic-optimizer', type=int, default=0, help='Whether to use hyperbolic optimizer')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

if args.dataset in unlabeled_datasets:
    dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, transform=Degree())
else:
    dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr)

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#torch.set_default_dtype(torch.float32)

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

acc = []
for i in range(10):
    model = WLHN(dataset.num_features, args.hidden_dim, args.depth, args.tau, args.classifier, dataset.num_classes, args.dropout).to(device)
    if (args.hyperbolic_optimizer == 1):
        optimizer = gt.optim.RiemannianAdam(model.parameters(), lr=args.lr, stabilize=5)
    else:
        print("outside")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_index = splits[i]['model_selection'][0]['train']
    val_index = splits[i]['model_selection'][0]['validation']
    test_index = splits[i]['test']

    test_dataset = dataset[test_index]
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print('---------------- Split {} ----------------'.format(i))

    best_val_loss, test_acc = 100, 0
    runtime = []
    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        if best_val_loss >= val_loss:
            test_acc = test(test_loader)
            best_val_loss = val_loss
        if epoch % 20 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                    epoch, train_loss, val_loss, test_acc))
        runtime.append(time.time()-start)
    print('Avg runtime per epoch:', np.mean(runtime))
    
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))

 