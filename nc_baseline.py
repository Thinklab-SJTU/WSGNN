import copy
import argparse

from utils.dataset import load_nc_dataset
from models.baseline import *
from models.metric import *
from parse import parser_add_main_args
from utils.split import *
from utils.negative_sampling import *
from utils.eval_utils import *


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset)
origin_dataset = copy.deepcopy(dataset)
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]
print(f"num nodes {n} | num classes {c} | num node feats {d}")

node_train_prop_list = [0.7, 0.3]
node_valid_prop = 0.1
seed = 76

for i, node_train_prop in enumerate(node_train_prop_list):

    fix_seed(seed)
    print(f'node_train_prop: {node_train_prop:.2f}, '
          f'seed: {seed:02d}')
    dataset = copy.deepcopy(origin_dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)
    node_split = RandomNodeSplit(train_prop=node_train_prop, valid_prop=node_valid_prop)
    dataset = node_split(dataset)
    dataset.edge_mask['train_mask'] = torch.ones_like(dataset.edge_mask['train_mask'])
    dataset.edge_mask['val_mask'] = torch.zeros_like(dataset.edge_mask['val_mask'])
    dataset.edge_mask['test_mask'] = torch.zeros_like(dataset.edge_mask['test_mask'])

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

    dataset.node_mask['train_mask'], dataset.node_mask['val_mask'], dataset.node_mask['test_mask'] = \
        dataset.node_mask['train_mask'].to(device), dataset.node_mask['val_mask'].to(device), \
        dataset.node_mask['test_mask'].to(device)

    dataset.edge_mask['train_mask'], dataset.edge_mask['val_mask'], dataset.edge_mask['test_mask'] = \
        dataset.edge_mask['train_mask'].to(device), dataset.edge_mask['val_mask'].to(device), \
        dataset.edge_mask['test_mask'].to(device)

    model = BaselineModel(args, n, c, d, dataset, device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_node_val = -np.Inf
    best_node_test = -np.Inf
    best_node_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output1, output2 = model(dataset, dataset.graph['edge_index'][:, dataset.edge_mask['train_mask']])
        output1 = F.log_softmax(output1, dim=1)
        output2 = output2.view(-1)
        loss = criterion(output1[dataset.node_mask['train_mask']], dataset.label.squeeze(1)[dataset.node_mask['train_mask']])
        loss.backward()
        optimizer.step()

        result = baseline_evaluate_nc_f1(model, dataset,
                                         dataset.graph['edge_index'][:, dataset.edge_mask['train_mask']])
        if result[1] >= best_node_val:
            best_node_val = result[1]
            best_node_test = result[2]
            best_node_epoch = epoch

        if (epoch + 1) % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'NC_Train_F1: {100 * result[0]:.2f}%, '
                  f'NC_Valid_F1: {100 * result[1]:.2f}%, '
                  f'NC_Test_F1: {100 * result[2]:.2f}%')
    print('\n')
    print(f'node_val: {100 * best_node_val:.2f}%, '
          f'node_test: {100 * best_node_test:.2f}%, '
          f'node_epoch: {best_node_epoch:02d}\n')
