from models.gnn import *
from models.graph_learner import *
from utils.constant import VERY_SMALL_NUMBER

from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class QModel(nn.Module):
    def __init__(self, args, d, n, c, device):
        super(QModel, self).__init__()
        self.device = device
        self.graph_skip_conn = args.graph_skip_conn

        self.encoder = Dense_APPNP_Net(in_channels=d,
                                       hidden_channels=args.hidden_channels,
                                       out_channels=c,
                                       dropout=args.dropout,
                                       K=args.hops,
                                       alpha=args.alpha).to(device)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=args.graph_learn_num_pers)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=args.graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.graph_learner1.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, init_adj=None):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

        return raw_adj, adj

    def forward(self, data):
        node_features = data.graph['node_feat']
        train_index = data.graph['edge_index'][:, data.edge_mask['train_mask']]
        edge_weight = None
        _edge_index, edge_weight = gcn_norm(
            train_index, edge_weight, data.graph['num_nodes'], False,
            dtype=node_features.dtype)
        row, col = _edge_index
        init_adj_sparse = SparseTensor(row=col, col=row, value=edge_weight,
                                       sparse_sizes=(data.graph['num_nodes'], data.graph['num_nodes']))
        init_adj = init_adj_sparse.to_dense()

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features, self.graph_skip_conn, init_adj)
        node_vec_1 = self.encoder(node_features, adj_1)

        node_vec_2 = self.encoder(node_features, init_adj)
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1),
                                            self.graph_skip_conn, init_adj)

        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class PModel(nn.Module):
    def __init__(self, args, d, n, c, device):
        super(PModel, self).__init__()
        self.device = device

        self.encoder1 = Dense_APPNP_Net(in_channels=d,
                                        hidden_channels=args.hidden_channels,
                                        out_channels=c,
                                        dropout=args.dropout,
                                        K=args.hops,
                                        alpha=args.alpha).to(device)

        self.encoder2 = MLP(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_mlp_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn).to(device)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=args.graph_learn_num_pers)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=args.graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()
        self.graph_learner.reset_parameters()
        self.graph_learner2.reset_parameters()

    def learn_graph(self, graph_learner, node_features):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

        return raw_adj, adj

    def forward(self, data):
        node_features = data.graph['node_feat']

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, node_features)
        node_vec_1 = self.encoder1(node_features, adj_1)

        node_vec_2 = self.encoder2(node_features)
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([node_vec_1, node_vec_2], dim=1))

        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        adj = 0.5 * adj_1 + 0.5 * adj_2

        return output, adj


class WSGNN(nn.Module):
    def __init__(self, args, d, n, c, device):
        super(WSGNN, self).__init__()
        self.P_Model = PModel(args, d, n, c, device)
        self.Q_Model = QModel(args, d, n, c, device)

    def reset_parameters(self):
        self.P_Model.reset_parameters()
        self.Q_Model.reset_parameters()

    def forward(self, data):
        q_y, q_a = self.Q_Model.forward(data)
        p_y, p_a = self.P_Model.forward(data)
        return p_y, p_a, q_y, q_a


class QModel_LP(nn.Module):
    """
    Qmodel for link prediction only task.
    """

    def __init__(self, args, d, n, c, device):
        super(QModel_LP, self).__init__()
        self.device = device
        self.graph_skip_conn = args.graph_skip_conn

        self.encoder = Dense_APPNP_Net(in_channels=d,
                                       hidden_channels=args.hidden_channels,
                                       out_channels=c,
                                       dropout=args.dropout,
                                       K=args.hops,
                                       alpha=args.alpha).to(device)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=args.graph_learn_num_pers).to(device)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=args.graph_learn_num_pers).to(device)

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, init_adj=None):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

        return raw_adj, adj

    def forward(self, data):
        features = data.graph['node_feat']
        train_index = data.graph['edge_index'][:, data.edge_mask['train_mask']]
        edge_weight = None
        _edge_index, edge_weight = gcn_norm(
            train_index, edge_weight, data.graph['num_nodes'], False,
            dtype=data.graph['node_feat'].dtype)
        row, col = _edge_index
        init_adj = SparseTensor(row=col, col=row, value=edge_weight,
                                sparse_sizes=(data.graph['num_nodes'], data.graph['num_nodes']))
        init_adj = init_adj.to_dense().float().to(train_index.device)

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, features, self.graph_skip_conn, init_adj)
        embedding_1 = self.encoder(features, adj_1)

        embedding_2 = self.encoder(features, init_adj)
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([embedding_1, embedding_2], dim=1), self.graph_skip_conn, init_adj)

        adj = raw_adj_2

        return adj


class PModel_LP(nn.Module):
    """
    Pmodel for link prediction only task.
    """

    def __init__(self, args, d, n, c, device):
        super(PModel_LP, self).__init__()
        self.device = device
        self.encoder1 = Dense_APPNP_Net(in_channels=d,
                                        hidden_channels=args.hidden_channels,
                                        out_channels=c,
                                        dropout=args.dropout,
                                        K=args.hops,
                                        alpha=args.alpha).to(device)

        self.encoder2 = MLP(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_mlp_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn).to(device)

        self.graph_learner1 = GraphLearner(input_size=d, num_pers=args.graph_learn_num_pers).to(device)
        self.graph_learner2 = GraphLearner(input_size=2 * c, num_pers=args.graph_learn_num_pers).to(device)

    def learn_graph(self, graph_learner, node_features):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        return raw_adj, adj

    def forward(self, data):
        features = data.graph['node_feat']

        raw_adj_1, adj_1 = self.learn_graph(self.graph_learner1, features)
        embedding_1 = self.encoder1(features, adj_1)

        embedding_2 = self.encoder2(features)
        raw_adj_2, adj_2 = self.learn_graph(self.graph_learner2, torch.cat([embedding_1, embedding_2], dim=1))
        adj = raw_adj_2

        return adj


class WSGNN_LP(nn.Module):
    def __init__(self, args, d, n, c, device):
        super(WSGNN_LP, self).__init__()
        self.P_Model = PModel_LP(args, d, n, c, device)
        self.Q_Model = QModel_LP(args, d, n, c, device)

    def reset_parameters(self):
        self.P_Model.reset_parameters()
        self.Q_Model.reset_parameters()

    def forward(self, data):
        q_a = self.Q_Model.forward(data)
        p_a = self.P_Model.forward(data)
        return p_a, q_a