def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--checkpoint-path", metavar="FILE", help="overload checkpoint path")

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--display_step', type=int, default=1, help='how often to print')

    parser.add_argument('--lambda1', type=float, default=1., help='weight of NC loss')
    parser.add_argument('--lambda2', type=float, default=1., help='weight of LP loss')

    parser.add_argument('--method', '-m', type=str, default='gcn')

    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of mlp layers')

    parser.add_argument('--graph_learn_num_pers', type=int, default=4)
    parser.add_argument('--graph_skip_conn', type=float, default=0.8)

    parser.add_argument('--neg_train_samples', action='store_true', help='add negative train samples')
    parser.add_argument('--neg_sampling_ratio', type=float, default=1.,
                        help='neg sampling ratio')














