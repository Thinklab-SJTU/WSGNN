import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_f1(y_true, y_pred, average='binary'):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()
    f1 = f1_score(y_pred, y_true, average=average)

    return f1


@torch.no_grad()
def baseline_evaluate_acc(model, dataset, edge_index,
                          num_edge_train, num_edge_valid, num_edge_test,
                          edge_label_train, edge_label_valid, edge_label_test):
    """
    :return accurary for node classification and roc_auc for link_prediction
    """

    model.eval()
    out1, out2 = model(dataset, edge_index)
    out2 = out2.view(-1)
    train_acc = eval_acc(dataset.label[dataset.node_mask['train_mask']], out1[dataset.node_mask['train_mask']])
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 out2[:num_edge_train].cpu().numpy())
    valid_acc = eval_acc(dataset.label[dataset.node_mask['val_mask']], out1[dataset.node_mask['val_mask']])
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 out2[num_edge_train:num_edge_train + num_edge_valid].cpu().numpy())
    test_acc = eval_acc(dataset.label[dataset.node_mask['test_mask']], out1[dataset.node_mask['test_mask']])
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                out2[num_edge_train + num_edge_valid:].cpu().numpy())
    return train_acc, train_rocauc, valid_acc, valid_rocauc, test_acc, test_rocauc


@torch.no_grad()
def baseline_evaluate_f1(model, dataset, edge_index,
                         num_edge_train, num_edge_valid, num_edge_test,
                         edge_label_train, edge_label_valid, edge_label_test):
    """
    :return f1 score for node classification and roc_auc for link_prediction
    """

    model.eval()
    out1, out2 = model(dataset, edge_index)
    out2 = out2.view(-1)
    train_f1 = eval_f1(dataset.label[dataset.node_mask['train_mask']], out1[dataset.node_mask['train_mask']])
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 out2[:num_edge_train].cpu().numpy())
    valid_f1 = eval_f1(dataset.label[dataset.node_mask['val_mask']], out1[dataset.node_mask['val_mask']])
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 out2[num_edge_train:num_edge_train + num_edge_valid].cpu().numpy())
    test_f1 = eval_f1(dataset.label[dataset.node_mask['test_mask']], out1[dataset.node_mask['test_mask']])
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                out2[num_edge_train + num_edge_valid:].cpu().numpy())
    return train_f1, train_rocauc, valid_f1, valid_rocauc, test_f1, test_rocauc


@torch.no_grad()
def baseline_evaluate_nc_f1(model, dataset, edge_index):
    '''
    :return f1 score for node classification
    '''

    model.eval()
    out1, _ = model(dataset, edge_index)
    train_f1 = eval_f1(dataset.label[dataset.node_mask['train_mask']], out1[dataset.node_mask['train_mask']])
    valid_f1 = eval_f1(dataset.label[dataset.node_mask['val_mask']], out1[dataset.node_mask['val_mask']])
    test_f1 = eval_f1(dataset.label[dataset.node_mask['test_mask']], out1[dataset.node_mask['test_mask']])
    return train_f1, valid_f1, test_f1


@torch.no_grad()
def baseline_evaluate_lp(model, dataset, edge_index,
                num_edge_train, num_edge_valid, num_edge_test,
                edge_label_train, edge_label_valid, edge_label_test):
    '''
    :return accurary for node classification and roc_auc for link_prediction
    '''

    model.eval()
    out1, out2 = model(dataset, edge_index)
    out2 = out2.view(-1)
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 out2[:num_edge_train].cpu().numpy())
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 out2[num_edge_train:num_edge_train + num_edge_valid].cpu().numpy())
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                out2[num_edge_train + num_edge_valid:].cpu().numpy())
    return train_rocauc, valid_rocauc, test_rocauc


@torch.no_grad()
def evaluate_acc(model, dataset, edge_index,
                 num_edge_train, num_edge_valid, num_edge_test,
                 edge_label_train, edge_label_valid, edge_label_test):
    '''
    :return accuracy for node classification and roc_auc for link_prediction
    '''

    model.eval()
    p_y, p_a, q_y, q_a = model(dataset)
    p_y = F.log_softmax(p_y, dim=1)
    q_y = F.log_softmax(q_y, dim=1)
    train_acc = eval_acc(dataset.label[dataset.node_mask['train_mask']], q_y[dataset.node_mask['train_mask']])
    valid_acc = eval_acc(dataset.label[dataset.node_mask['val_mask']], q_y[dataset.node_mask['val_mask']])
    test_acc = eval_acc(dataset.label[dataset.node_mask['test_mask']], q_y[dataset.node_mask['test_mask']])
    q_a = q_a.view(-1)
    edge_index_new = edge_index[0] * dataset.graph['num_nodes'] + edge_index[1]
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 q_a[edge_index_new[:num_edge_train]].cpu().numpy())
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 q_a[edge_index_new[num_edge_train:num_edge_train + num_edge_valid]].cpu().numpy())
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                q_a[edge_index_new[num_edge_train + num_edge_valid:]].cpu().numpy())

    return train_acc, train_rocauc, valid_acc, valid_rocauc, test_acc, test_rocauc


@torch.no_grad()
def evaluate_f1(model, dataset, edge_index,
                num_edge_train, num_edge_valid, num_edge_test,
                edge_label_train, edge_label_valid, edge_label_test):
    '''
    :return f1 score for node classification and roc_auc for link_prediction
    '''

    model.eval()
    p_y, p_a, q_y, q_a = model(dataset)
    p_y = F.log_softmax(p_y, dim=1)
    q_y = F.log_softmax(q_y, dim=1)
    train_f1 = eval_f1(dataset.label[dataset.node_mask['train_mask']], q_y[dataset.node_mask['train_mask']])
    valid_f1 = eval_f1(dataset.label[dataset.node_mask['val_mask']], q_y[dataset.node_mask['val_mask']])
    test_f1 = eval_f1(dataset.label[dataset.node_mask['test_mask']], q_y[dataset.node_mask['test_mask']])
    q_a = q_a.view(-1)
    edge_index_new = edge_index[0] * dataset.graph['num_nodes'] + edge_index[1]
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 q_a[edge_index_new[:num_edge_train]].cpu().numpy())
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 q_a[edge_index_new[num_edge_train:num_edge_train + num_edge_valid]].cpu().numpy())
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                q_a[edge_index_new[num_edge_train + num_edge_valid:]].cpu().numpy())

    return train_f1, train_rocauc, valid_f1, valid_rocauc, test_f1, test_rocauc


@torch.no_grad()
def evaluate_nc_acc(model, dataset):
    """
    :return accuracy for node classification
    """

    model.eval()
    _, _, q_y, _ = model(dataset)
    q_y = F.log_softmax(q_y, dim=1)
    train_acc = eval_acc(dataset.label[dataset.node_mask['train_mask']], q_y[dataset.node_mask['train_mask']])
    valid_acc = eval_acc(dataset.label[dataset.node_mask['val_mask']], q_y[dataset.node_mask['val_mask']])
    test_acc = eval_acc(dataset.label[dataset.node_mask['test_mask']], q_y[dataset.node_mask['test_mask']])

    return train_acc, valid_acc, test_acc


def evaluate_nc_f1(model, dataset):
    """
    :return f1 score for node classification
    """

    model.eval()
    _, _, q_y, _ = model(dataset)
    q_y = F.log_softmax(q_y, dim=1)
    train_f1 = eval_f1(dataset.label[dataset.node_mask['train_mask']], q_y[dataset.node_mask['train_mask']])
    valid_f1 = eval_f1(dataset.label[dataset.node_mask['val_mask']], q_y[dataset.node_mask['val_mask']])
    test_f1 = eval_f1(dataset.label[dataset.node_mask['test_mask']], q_y[dataset.node_mask['test_mask']])

    return train_f1, valid_f1, test_f1


@torch.no_grad()
def evaluate_lp(model, dataset, edge_index,
                num_edge_train, num_edge_valid, num_edge_test,
                edge_label_train, edge_label_valid, edge_label_test):
    """
    :return roc_auc for link_prediction
    """

    model.eval()
    p_a, q_a = model(dataset)
    q_a = q_a.view(-1)
    edge_index_new = edge_index[0] * dataset.graph['num_nodes'] + edge_index[1]
    train_rocauc = roc_auc_score(edge_label_train.cpu().numpy(),
                                 q_a[edge_index_new[:num_edge_train]].cpu().numpy())
    valid_rocauc = roc_auc_score(edge_label_valid.cpu().numpy(),
                                 q_a[edge_index_new[num_edge_train:num_edge_train + num_edge_valid]].cpu().numpy())
    test_rocauc = roc_auc_score(edge_label_test.cpu().numpy(),
                                q_a[edge_index_new[num_edge_train + num_edge_valid:]].cpu().numpy())

    return train_rocauc, valid_rocauc, test_rocauc

