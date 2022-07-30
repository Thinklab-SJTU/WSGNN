import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, APPNP


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, use_bn=False):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
        return x


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCNEncoder, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs):
            x = conv(x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, out_heads=1, use_bn=False):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, dropout=dropout, heads=out_heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, heads=2, use_bn=False):
        super(GATEncoder, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 1):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, data.graph['edge_index'][:, data.edge_mask['train_mask']])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.graph['node_feat'], data.graph['edge_index'][:, data.edge_mask['train_mask']]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


class DenseAPPNP(nn.Module):
    def __init__(self, K, alpha):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, x, adj_t):
        h = x
        for k in range(self.K):
            if adj_t.is_sparse:
                x = torch_sparse.spmm(adj_t, x)
            else:
                x = torch.matmul(adj_t, x)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        return x


class Dense_APPNP_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=.5, K=10, alpha=.1):
        super(Dense_APPNP_Encoder, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.prop1 = DenseAPPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x, adj_t):
        x = self.lin1(x)
        x = self.prop1(x, adj_t)
        return x


class Dense_APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.5, K=10, alpha=.1):
        super(Dense_APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = DenseAPPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, adj_t)
        return x