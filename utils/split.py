"""
Reference: pyg-random_link_split
"""
from utils.dataset import NCDataset
from torch_geometric.utils import add_self_loops
from utils.negative_sampling import *


class RandomNodeSplit:
    def __init__(
            self,
            train_prop: float = .5,
            valid_prop: float = .25,
    ):
        self.train_prop = train_prop
        self.valid_prop = valid_prop

    def __call__(self, data: NCDataset) -> NCDataset:
        data.node_mask['train_mask'] = torch.zeros_like(data.node_mask['train_mask'])
        data.node_mask['val_mask'] = torch.zeros_like(data.node_mask['val_mask'])
        data.node_mask['test_mask'] = torch.zeros_like(data.node_mask['test_mask'])

        perm = torch.randperm(data.graph['num_nodes'], device=data.graph['node_feat'].device)
        num_nodes = data.graph['num_nodes']
        num_train = int(num_nodes * self.train_prop)
        num_valid = int(num_nodes * self.valid_prop)

        train_nodes = perm[:num_train]
        valid_nodes = perm[num_train:num_train + num_valid]
        test_nodes = perm[num_train + num_valid:]

        data.node_mask['train_mask'][train_nodes] = 1
        data.node_mask['val_mask'][valid_nodes] = 1
        data.node_mask['test_mask'][test_nodes] = 1

        return data


class RandomNodeSplit2:
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
    ):
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid

    def __call__(self, data: NCDataset) -> NCDataset:
        data.node_mask['train_mask'] = torch.zeros_like(data.node_mask['train_mask'])
        data.node_mask['val_mask'] = torch.zeros_like(data.node_mask['val_mask'])
        data.node_mask['test_mask'] = torch.zeros_like(data.node_mask['test_mask'])

        perm = torch.randperm(data.graph['num_nodes'], device=data.graph['node_feat'].device)
        train_cnt = np.zeros(data.label.max().item() + 1, dtype=np.int)

        for i in range(perm.numel()):
            label = data.label[perm[i]]
            if train_cnt[label] < self.num_labels_per_class:
                train_cnt[label] += 1
                data.node_mask['train_mask'][perm[i]] = 1
            elif data.node_mask['val_mask'].sum() < self.num_valid:
                data.node_mask['val_mask'][perm[i]] = 1
            else:
                data.node_mask['test_mask'][perm[i]] = 1

        return data


class RandomLinkSplit:
    def __init__(
            self,
            train_prop: float = .5,
            valid_prop: float = .25,
            is_undirected: bool = False,
            neg_sampling_ratio: float = 1
    ):
        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.is_undirected = is_undirected
        self.neg_sampling_ratio = neg_sampling_ratio

    def __call__(self, data: NCDataset) -> NCDataset:
        if self.is_undirected:
            edge_index = data.graph['edge_index'][:, data.graph['edge_index'][0] <= data.graph['edge_index'][1]]
            num_edges = data.graph['num_edges'] // 2

            num_train = int(num_edges * self.train_prop)
            num_valid = int(num_edges * self.valid_prop)
            num_test = num_edges - num_train - num_valid

            perm = torch.randperm(num_edges, device=data.graph['edge_index'].device)

            train_edges = perm[:num_train]
            valid_edges = perm[num_train:num_train + num_valid]
            test_edges = perm[num_train + num_valid:]

            edge_index_train = edge_index[:, train_edges]
            edge_index_valid = edge_index[:, valid_edges]
            edge_index_test = edge_index[:, test_edges]

            num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_valid = int(num_valid * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)
            num_neg = num_neg_train + num_neg_valid + num_neg_test

            edge_index_train = torch.cat([edge_index_train, edge_index_train.flip([0])], dim=-1)
            edge_index_valid = torch.cat([edge_index_valid, edge_index_valid.flip([0])], dim=-1)
            edge_index_test = torch.cat([edge_index_test, edge_index_test.flip([0])], dim=-1)

            data.graph['edge_index'] = torch.cat([edge_index_train, edge_index_valid, edge_index_test], dim=-1)

            data.edge_mask['train_mask'] = torch.zeros_like(data.edge_mask['train_mask'])
            data.edge_mask['val_mask'] = torch.zeros_like(data.edge_mask['val_mask'])
            data.edge_mask['test_mask'] = torch.zeros_like(data.edge_mask['test_mask'])

            data.edge_mask['train_mask'][:2 * num_train, ] = 1
            data.edge_mask['val_mask'][2 * num_train:2 * (num_train + num_valid), ] = 1
            data.edge_mask['test_mask'][2 * (num_train + num_valid):, ] = 1

            neg_edge_index = negative_sampling(
                add_self_loops(data.graph['edge_index'])[0], num_nodes=data.graph['num_nodes'],
                num_neg_samples=2 * num_neg, method='sparse', force_undirected=True)

            neg_edge_index_train = torch.cat([neg_edge_index[:, :num_neg_train],
                                              neg_edge_index[:, num_neg:num_neg + num_neg_train]], dim=-1)
            neg_edge_index_valid = torch.cat([neg_edge_index[:, num_neg_train:num_neg_train + num_neg_valid],
                                              neg_edge_index[:, num_neg + num_neg_train:
                                                             num_neg + num_neg_train + num_neg_valid]], dim=-1)
            neg_edge_index_test = torch.cat([neg_edge_index[:, num_neg_train + num_neg_valid:num_neg],
                                             neg_edge_index[:, num_neg + num_neg_train + num_neg_valid:]], dim=-1)
            data.graph['neg_edge_index'] = torch.cat([neg_edge_index_train,
                                                      neg_edge_index_valid,
                                                      neg_edge_index_test], dim=-1)

            data.neg_edge_mask['train_mask'] = torch.zeros(2 * num_neg).bool()
            data.neg_edge_mask['train_mask'][:2 * num_neg_train, ] = 1
            data.neg_edge_mask['val_mask'] = torch.zeros(2 * num_neg).bool()
            data.neg_edge_mask['val_mask'][2 * num_neg_train:2 * (num_neg_train + num_neg_valid), ] = 1
            data.neg_edge_mask['test_mask'] = torch.zeros(2 * num_neg).bool()
            data.neg_edge_mask['test_mask'][2 * (num_neg_train + num_neg_valid):, ] = 1

        else:
            edge_index = data.graph['edge_index']
            num_edges = data.graph['num_edges']

            num_train = int(num_edges * self.train_prop)
            num_valid = int(num_edges * self.valid_prop)
            num_test = num_edges - num_train - num_valid

            num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_valid = int(num_valid * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)
            num_neg = num_neg_train + num_neg_valid + num_neg_test

            perm = torch.randperm(data.graph['num_edges'], device=data.graph['edge_index'].device)

            train_edges = perm[:num_train]
            valid_edges = perm[num_train:num_train + num_valid]
            test_edges = perm[num_train + num_valid:]

            edge_index_train = edge_index[:, train_edges]
            edge_index_valid = edge_index[:, valid_edges]
            edge_index_test = edge_index[:, test_edges]

            data.graph['edge_index'] = torch.cat([edge_index_train, edge_index_valid, edge_index_test], dim=-1)

            data.edge_mask['train_mask'][:num_train, ] = 1
            data.edge_mask['val_mask'][num_train:num_train + num_valid, ] = 1
            data.edge_mask['test_mask'][num_train + num_valid:, ] = 1

            neg_edge_index = negative_sampling(
                add_self_loops(data.graph['edge_index'])[0], num_nodes=data.graph['num_nodes'],
                num_neg_samples=num_neg, method='sparse')

            neg_edge_index_train = neg_edge_index[:, :num_neg_train]
            neg_edge_index_valid = neg_edge_index[:, num_neg_train:num_neg_train + num_neg_valid]
            neg_edge_index_test = neg_edge_index[:, num_neg_train + num_neg_valid:]
            data.graph['neg_edge_index'] = torch.cat([neg_edge_index_train,
                                                      neg_edge_index_valid,
                                                      neg_edge_index_test], dim=-1)
            data.neg_edge_mask['train_mask'] = torch.zeros(num_neg).bool()
            data.neg_edge_mask['train_mask'][:num_neg_train, ] = 1
            data.neg_edge_mask['val_mask'] = torch.zeros(num_neg).bool()
            data.neg_edge_mask['val_mask'][num_neg_train:num_neg_train + num_neg_valid, ] = 1
            data.neg_edge_mask['test_mask'] = torch.zeros(num_neg).bool()
            data.neg_edge_mask['test_mask'][num_neg_train + num_neg_valid:, ] = 1

        return data
