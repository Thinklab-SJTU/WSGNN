import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLoss(nn.Module):
    def __init__(self, lambda1, lambda2, weight):
        super(BaselineLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss1 = nn.NLLLoss()
        self.loss2 = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, data, output1, label1, output2, label2):
        loss = self.lambda1 * self.loss1(output1[data.node_mask['train_mask']], label1[data.node_mask['train_mask']]) \
              + self.lambda2 * self.loss2(output2, label2)

        return loss


class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()

    def forward(self, data, log_p_y, p_a, log_q_y, q_a, edge_index, edge_label, weight):
        y_obs = data.label.squeeze(1)[data.node_mask['train_mask']]
        log_p_y_obs = log_p_y[data.node_mask['train_mask']]
        p_y_obs = torch.exp(log_p_y_obs)
        log_p_y_miss = log_p_y[data.node_mask['train_mask'] == 0]
        p_y_miss = torch.exp(log_p_y_miss)
        log_q_y_obs = log_q_y[data.node_mask['train_mask']]
        q_y_obs = torch.exp(log_q_y_obs)
        log_q_y_miss = log_q_y[data.node_mask['train_mask'] == 0]
        q_y_miss = torch.exp(log_q_y_miss)

        loss_p_y = 1 * F.nll_loss(log_p_y_obs, y_obs) - torch.mean(q_y_miss * log_p_y_miss)
        loss_q_y = torch.mean(q_y_miss * log_q_y_miss)

        p_a_new = p_a.view(-1)
        q_a_new = q_a.view(-1)

        edge_index_new = edge_index[0] * data.graph['num_nodes'] + edge_index[1]
        mask = torch.zeros_like(q_a_new)
        mask[edge_index_new] = 1

        p_a_new = p_a_new.unsqueeze(1)
        q_a_new = q_a_new.unsqueeze(1)

        p_a_new = torch.cat([torch.zeros_like(p_a_new), p_a_new], dim=1)
        log_p_a_new = F.log_softmax(p_a_new, dim=1)

        q_a_new = torch.cat([torch.zeros_like(q_a_new), q_a_new], dim=1)
        log_q_a_new = F.log_softmax(q_a_new, dim=1)

        log_p_a_obs = log_p_a_new[edge_index_new]
        log_p_a_miss = log_p_a_new[mask == 0]
        p_a_obs = torch.exp(log_p_a_obs)
        p_a_miss = torch.exp(log_p_a_miss)

        log_q_a_obs = log_q_a_new[edge_index_new]
        log_q_a_miss = log_q_a_new[mask == 0]
        q_a_obs = torch.exp(log_q_a_obs)
        q_a_miss = torch.exp(log_q_a_miss)

        loss_p_a = F.nll_loss(log_p_a_obs, edge_label, weight=weight) - torch.mean(log_p_a_miss * q_a_miss)
        loss_q_a = torch.mean(q_a_miss * log_q_a_miss)

        loss_y_obs = 10 * F.nll_loss(log_q_y_obs, y_obs)
        loss_a_obs = 10 * F.nll_loss(log_q_a_obs, edge_label, weight=weight)

        loss = loss_p_y + loss_q_y + loss_p_a + loss_q_a + loss_y_obs + loss_a_obs

        return loss


class ELBONCLoss(nn.Module):
    def __init__(self):
        super(ELBONCLoss, self).__init__()

    def forward(self, data, log_p_y, log_q_y):
        y_obs = data.label.squeeze(1)[data.node_mask['train_mask']]
        log_p_y_obs = log_p_y[data.node_mask['train_mask']]
        p_y_obs = torch.exp(log_p_y_obs)
        log_p_y_miss = log_p_y[data.node_mask['train_mask'] == 0]
        p_y_miss = torch.exp(log_p_y_miss)
        log_q_y_obs = log_q_y[data.node_mask['train_mask']]
        q_y_obs = torch.exp(log_q_y_obs)
        log_q_y_miss = log_q_y[data.node_mask['train_mask'] == 0]
        q_y_miss = torch.exp(log_q_y_miss)

        loss_p_y = F.nll_loss(log_p_y_obs, y_obs) - torch.mean(q_y_miss * log_p_y_miss)
        loss_q_y = torch.mean(q_y_miss * log_q_y_miss)

        loss_y_obs = 10 * F.nll_loss(log_q_y_obs, y_obs)

        loss = loss_p_y + loss_q_y + loss_y_obs

        return loss


class ELBOLPLoss(nn.Module):
    def __init__(self):
        super(ELBOLPLoss, self).__init__()

    def forward(self, data, p_a, q_a, edge_index, edge_label, weight):
        p_a_new = p_a.view(-1)
        q_a_new = q_a.view(-1)

        edge_index_new = edge_index[0] * data.graph['num_nodes'] + edge_index[1]
        mask = torch.zeros_like(q_a_new)
        mask[edge_index_new] = 1

        p_a_new = p_a_new.unsqueeze(1)
        q_a_new = q_a_new.unsqueeze(1)

        p_a_new = torch.cat([torch.zeros_like(p_a_new), p_a_new], dim=1)
        log_p_a_new = F.log_softmax(p_a_new, dim=1)

        q_a_new = torch.cat([torch.zeros_like(q_a_new), q_a_new], dim=1)
        log_q_a_new = F.log_softmax(q_a_new, dim=1)

        log_p_a_obs = log_p_a_new[edge_index_new]
        log_p_a_miss = log_p_a_new[mask == 0]
        p_a_obs = torch.exp(log_p_a_obs)
        p_a_miss = torch.exp(log_p_a_miss)

        log_q_a_obs = log_q_a_new[edge_index_new]
        log_q_a_miss = log_q_a_new[mask == 0]
        q_a_obs = torch.exp(log_q_a_obs)
        q_a_miss = torch.exp(log_q_a_miss)

        loss_p_a = F.nll_loss(log_p_a_obs, edge_label, weight=weight) - torch.mean(log_p_a_miss * q_a_miss)
        loss_q_a = torch.mean(q_a_miss * log_q_a_miss)

        loss_a_obs = 10 * F.nll_loss(log_q_a_obs, edge_label, weight=weight)
        loss = loss_p_a + loss_q_a + loss_a_obs

        return loss
