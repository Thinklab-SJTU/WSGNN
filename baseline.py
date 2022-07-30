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

print(f"num nodes {n} | num classes {c} | num node feats {d}\n")

weight = [args.neg_sampling_ratio]
weight = torch.tensor(weight).to(device)

num_labels_per_class_list = [20, 10]
num_valid = 500
edge_train_prop_list = [0.5, 0.1, 0.01]
edge_valid_prop = 0.1
seed = 76

for i, num_labels_per_class in enumerate(num_labels_per_class_list):
    for j, edge_train_prop in enumerate(edge_train_prop_list):

        fix_seed(seed)
        print(f'num_labels_per_class: {num_labels_per_class:02d}, '
              f'edge_train_prop: {edge_train_prop:.2f}, '
              f'seed: {seed:02d}')

        dataset = copy.deepcopy(origin_dataset)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        dataset.label = dataset.label.to(device)
        node_split = RandomNodeSplit2(num_labels_per_class=num_labels_per_class,
                                      num_valid=num_valid)
        dataset = node_split(dataset)
        link_split = RandomLinkSplit(train_prop=edge_train_prop,
                                     valid_prop=edge_valid_prop,
                                     is_undirected=not args.directed,
                                     neg_sampling_ratio=args.neg_sampling_ratio)
        dataset = link_split(dataset)

        dataset.graph['edge_index'], dataset.graph['node_feat'], dataset.graph['neg_edge_index'] = \
            dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device), \
            dataset.graph['neg_edge_index'].to(device)

        dataset.node_mask['train_mask'], dataset.node_mask['val_mask'], dataset.node_mask['test_mask'] = \
            dataset.node_mask['train_mask'].to(device), dataset.node_mask['val_mask'].to(device), \
            dataset.node_mask['test_mask'].to(device)

        dataset.edge_mask['train_mask'], dataset.edge_mask['val_mask'], dataset.edge_mask['test_mask'] = \
            dataset.edge_mask['train_mask'].to(device), dataset.edge_mask['val_mask'].to(device), \
            dataset.edge_mask['test_mask'].to(device)

        dataset.neg_edge_mask['train_mask'], dataset.neg_edge_mask['val_mask'], \
        dataset.neg_edge_mask['test_mask'] = dataset.neg_edge_mask['train_mask'].to(device), \
                                             dataset.neg_edge_mask['val_mask'].to(device), dataset.neg_edge_mask[
                                                 'test_mask'].to(device)

        num_pos_edge_train = dataset.graph['edge_index'][:, dataset.edge_mask['train_mask']].size(1)
        num_neg_edge_train = dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['train_mask']].size(1)
        num_edge_train = num_pos_edge_train + num_neg_edge_train
        num_pos_edge_valid = dataset.graph['edge_index'][:, dataset.edge_mask['val_mask']].size(1)
        num_neg_edge_valid = dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['val_mask']].size(1)
        num_edge_valid = num_pos_edge_valid + num_neg_edge_valid
        num_pos_edge_test = dataset.graph['edge_index'][:, dataset.edge_mask['test_mask']].size(1)
        num_neg_edge_test = dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['test_mask']].size(1)
        num_edge_test = num_pos_edge_test + num_neg_edge_test

        edge_label_train = torch.cat([
            torch.ones(num_pos_edge_train, device=device),
            torch.zeros(num_neg_edge_train, device=device)
        ], dim=0)

        edge_index_valid = torch.cat([
            dataset.graph['edge_index'][:, dataset.edge_mask['val_mask']],
            dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['val_mask']]
        ], dim=-1)
        edge_label_valid = torch.cat([
            torch.ones(num_pos_edge_valid, device=device),
            torch.zeros(num_neg_edge_valid, device=device)
        ], dim=0)

        edge_index_test = torch.cat([
            dataset.graph['edge_index'][:, dataset.edge_mask['test_mask']],
            dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['test_mask']]
        ], dim=-1)
        edge_label_test = torch.cat([
            torch.ones(num_pos_edge_test, device=device),
            torch.zeros(num_neg_edge_test, device=device)
        ], dim=0)

        # Load method
        model = BaselineModel(args, n, c, d, dataset, device)
        criterion = BaselineLoss(lambda1=args.lambda1, lambda2=args.lambda2, weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_node_val = -np.Inf
        best_node_test = -np.Inf
        best_node_epoch = -1
        best_edge_val = -np.Inf
        best_edge_test = -np.Inf
        best_edge_epoch = -1

        for epoch in range(args.epochs):
            neg_edge_index_train = negative_sampling(
                edge_index=add_self_loops(torch.cat([
                    dataset.graph['edge_index'],
                    dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['val_mask']],
                    dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['test_mask']]
                ], dim=-1))[0],
                num_nodes=dataset.graph['num_nodes'],
                num_neg_samples=num_neg_edge_train, method='sparse', force_undirected=True).to(device)
            dataset.graph['neg_edge_index'][:, dataset.neg_edge_mask['train_mask']] = neg_edge_index_train
            edge_index_train = torch.cat([
                dataset.graph['edge_index'][:, dataset.edge_mask['train_mask']],
                neg_edge_index_train
            ], dim=-1)
            edge_index = torch.cat([edge_index_train, edge_index_valid, edge_index_test], dim=-1)

            model.train()
            optimizer.zero_grad()
            output1, output2 = model(dataset, edge_index)
            output1 = F.log_softmax(output1, dim=1)
            output2 = output2.view(-1)
            loss = criterion(dataset, output1,
                             dataset.label.squeeze(1),
                             output2[:num_edge_train],
                             edge_label_train)

            loss.backward()
            optimizer.step()

            result = baseline_evaluate_acc(model, dataset, edge_index,
                                           num_edge_train, num_edge_valid, num_edge_test,
                                           edge_label_train, edge_label_valid, edge_label_test)

            if result[2] >= best_node_val:
                best_node_val = result[2]
                best_node_test = result[4]
                best_node_epoch = epoch

            if result[3] >= best_edge_val:
                best_edge_val = result[3]
                best_edge_test = result[5]
                best_edge_epoch = epoch

            if (epoch + 1) % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'NC_Train_Acc: {100 * result[0]:.2f}%, '
                      f'NC_Valid_Acc: {100 * result[2]:.2f}%, '
                      f'NC_Test_Acc: {100 * result[4]:.2f}%, '
                      f'LP_Train_Roc: {result[1]:.4f}, '
                      f'LP_Valid_Roc: {result[3]:.4f}, '
                      f'LP_Test_Roc: {result[5]:.4f}')
        print('\n')
        print(f'node_val: {100 * best_node_val:.2f}%, '
              f'node_test: {100 * best_node_test:.2f}%, '
              f'node_epoch: {best_node_epoch:02d}, '
              f'edge_val: {best_edge_val:.4f}, '
              f'edge_test: {best_edge_test:.4f}, '
              f'edge_epoch: {best_edge_epoch:02d}\n')








