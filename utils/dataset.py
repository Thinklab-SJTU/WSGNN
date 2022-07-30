import os
import numpy as np
import torch
import torch_geometric.transforms as transform

from utils.load_data import load_synthetic
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid
import scipy.sparse as sp


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class NCDataset(object):
    def __init__(self, name):
        self.name = name  # original name
        self.graph = {}
        self.node_mask = {}
        self.edge_mask = {}
        self.neg_edge_mask = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname):
    """ Loader for NCDataset 
        Returns NCDataset 
    """
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_ccp_dataset(dataname)
    elif dataname in ('disease_nc', 'disease_lp'):
        dataset = load_synthetic_data(dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_graph(dataset, label, edge_index, node_feat, num_nodes, num_edges):
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'neg_edge_index': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes,
                     'num_edges': num_edges}
    dataset.label = torch.tensor(label)

    # add node mask
    node_train_mask = np.zeros((num_nodes,), dtype=int)
    node_val_mask = np.zeros((num_nodes,), dtype=int)
    node_test_mask = np.zeros((num_nodes,), dtype=int)

    dataset.node_mask = {'train_mask': torch.BoolTensor(node_train_mask),
                         'val_mask': torch.BoolTensor(node_val_mask),
                         'test_mask': torch.BoolTensor(node_test_mask)}

    # add edge mask
    edge_train_mask = np.zeros((num_edges,), dtype=int)
    edge_val_mask = np.zeros((num_edges,), dtype=int)
    edge_test_mask = np.zeros((num_edges,), dtype=int)

    dataset.edge_mask = {'train_mask': torch.BoolTensor(edge_train_mask),
                         'val_mask': torch.BoolTensor(edge_val_mask),
                         'test_mask': torch.BoolTensor(edge_test_mask)}

    dataset.neg_edge_mask = {'train_mask': None,
                             'val_mask': None,
                             'test_mask': None}

    return dataset


def load_ccp_dataset(filename):  # Cora/Citeseer/Pubmed
    path = os.path.join("data", filename)
    assert filename in ["cora", "pubmed", "citeseer"]
    origin_dataset = Planetoid(
        root=path, name=filename, transform=transform.NormalizeFeatures())
    data = origin_dataset[0]
    dataset = NCDataset(filename)
    label = data.y.numpy()
    edge_index = data.edge_index
    node_feat = data.x
    num_nodes = data.y.size(0)
    num_edges = edge_index.size(1)
    return load_graph(dataset, label, edge_index, node_feat, num_nodes, num_edges)


def load_synthetic_data(filename):
    A, label, features = load_synthetic(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    num_edges = edge_index.size(1)
    return load_graph(dataset, label, edge_index, node_feat, num_nodes, num_edges)
