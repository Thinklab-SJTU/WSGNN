import scipy.io
import numpy as np
import scipy.sparse
import torch
from os import path


def load_synthetic(filename):
    filepath = f"data/{filename}"
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(path.join(filepath, "{}.edges.csv".format(filename)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = scipy.sparse.load_npz(path.join(filepath, "{}.feats.npz".format(filename)))
    labels = np.load(path.join(filepath, "{}.labels.npy".format(filename))).astype(int)
    adj = scipy.sparse.csr_matrix(adj)
    adj, features = process(adj, features, False, False)
    return scipy.sparse.csr_matrix(adj), labels, features


def process(adj, features, normalize_adj, normalize_feats):
    if scipy.sparse.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    if normalize_adj:
        adj = normalize(adj + scipy.sparse.eye(adj.shape[0]))
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
