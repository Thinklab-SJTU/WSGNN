from models.gnn import *


class OutputLayer(nn.Module):
    def __init__(self, method, in_channels, out_channels, num_heads, dropout, hops, alpha):
        super(OutputLayer, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.method = method
        if method == 'gcn':
            self.node_decoder = GCNConv(in_channels, out_channels)
        elif method == 'gat':
            self.node_decoder = GATConv(in_channels, out_channels, num_heads, dropout=dropout, concat=False)
        elif method == 'appnp':
            self.node_decoder = APPNP(1, alpha=alpha)

    def reset_parameters(self):
        self.node_decoder.reset_parameters()

    def forward(self, z, edge_index, data):

        output1 = self.node_decoder(z, data.graph['edge_index'][:, data.edge_mask['train_mask']])
        output2 = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        return output1, output2


class BaselineModel(nn.Module):
    def __init__(self, args, n, c, d, dataset, device):
        super(BaselineModel, self).__init__()
        input_channel = args.hidden_channels
        if args.method == 'gcn':
            self.encoder = GCNEncoder(in_channels=d,
                                      hidden_channels=args.hidden_channels,
                                      num_layers=args.num_layers,
                                      dropout=args.dropout,
                                      use_bn=not args.no_bn).to(device)
        elif args.method == 'gat':
            self.encoder = GATEncoder(in_channels=d,
                                      hidden_channels=args.hidden_channels,
                                      num_layers=args.num_layers,
                                      dropout=args.dropout,
                                      heads=args.gat_heads,
                                      use_bn=not args.no_bn).to(device)
            input_channel *= args.gat_heads
        elif args.method == 'appnp':
            self.encoder = APPNP_Net(in_channels=d,
                                     hidden_channels=args.hidden_channels,
                                     out_channels=c,
                                     dropout=args.dropout,
                                     K=args.hops,
                                     alpha=args.alpha).to(device)
        self.decoder = OutputLayer(args.method, input_channel, c, args.out_heads, args.dropout, args.hops,
                                   args.alpha).to(device)
        self.method = args.method

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data, edge_index):
        z = self.encoder(data)
        output1, output2 = self.decoder(z, edge_index, data)
        return output1, output2
