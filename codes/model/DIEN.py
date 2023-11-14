# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from codes.utils.functions import get_dataset_summary_dict
from .utils.layers import DNN, AttentionSequencePoolingLayer, DynamicGRU


class PredictionLayer(nn.Module):

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


class InterestExtractor(nn.Module):
    def __init__(self, input_size, use_neg=False, init_std=0.001):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], 'sigmoid', init_std=init_std)
            # self.auxiliary_net = nn.Sequential(DNN(input_size * 2, [100, 50], 'relu', init_std=init_std),
            #                                    nn.Linear(50, 1))
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, keys, keys_length, neg_keys=None):

        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)  # [batch_size,dim]
        aux_loss = torch.zeros((1,), device=keys.device)  # [1,]

        # create zero mask for keys_length, to make sure 'pack_padded_sequence' safe
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]

        # batch_size validation check
        if masked_keys_length.shape[0] == 0:
            return zero_outputs,
        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
        packed_keys = pack_padded_sequence(masked_keys, lengths=masked_keys_length.cpu(), batch_first=True,
                                           enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                           total_length=max_length)

        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(
                interests[:, :-1, :],
                masked_keys[:, 1:, :],
                masked_neg_keys[:, 1:, :],
                masked_keys_length - 1)

        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        # keys_length >= 1
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)

        _, max_seq_length, embedding_size = states.size()
        # [new_batch,max_len-1,dim]
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        # [new_batch,max_len-1,dim]
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        # [new_batch,max_len-1,dim]
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length,
                                                                                       embedding_size)
        batch_size = states.size()[0]
        # [new_batch,max_len-1]
        mask = (torch.arange(max_seq_length, device=states.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1)).float()
        click_input = torch.cat([states, click_seq], dim=-1)  # [new_batch,max_len-1,2*dim]
        noclick_input = torch.cat([states, noclick_seq], dim=-1)  # [new_batch,max_len-1,2*dim]
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(click_input.view(
            batch_size * max_seq_length, embedding_size), use_last_activation=False).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(noclick_input.view(
            batch_size * max_seq_length, embedding_size), use_last_activation=False).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self,
                 input_size,
                 gru_type='GRU',
                 use_neg=False,
                 init_std=0.001,
                 att_hidden_size=(64, 16),
                 att_activation='sigmoid',
                 att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size,
                                                 gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        batch_size, dim = query.size()  # Target Item
        max_length = keys.size()[1]

        # check batch validation
        # [batch,dim]
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length, batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [b, H]
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [batch,max_length]
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length)  # [b, H]
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs.float()
        return zero_outputs


class DIEN(nn.Module):

    def __init__(self, dataset_summary_file, embed_size, sparse_len, dense_len,
                 seed, hist_itemfeature_dict_embedsize,
                 gru_type="AUGRU", use_negsampling=False, use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu',
                 att_hidden_units=(64, 16), att_activation="relu", att_weight_normalization=True,
                 l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001
                 ):
        super(DIEN, self).__init__()

        self.embed_size = embed_size
        self.dataset_summary = get_dataset_summary_dict(dataset_summary_file)
        self.embedding_features = nn.Embedding(num_embeddings=self.dataset_summary['feat_num'],
                                               embedding_dim=self.embed_size)

        self.use_negsampling = use_negsampling
        # structure: embedding layer -> interest extractor layer -> interest evolution layer -> DNN layer -> out

        # embedding layer
        # inherit -> self.embedding_dict
        input_size = sum([i for i in hist_itemfeature_dict_embedsize.values()])
        # interest extractor layer
        self.interest_extractor = InterestExtractor(input_size=input_size, use_neg=use_negsampling, init_std=init_std)
        # interest evolution layer
        self.interest_evolution = InterestEvolving(
            input_size=input_size,
            gru_type=gru_type,
            use_neg=use_negsampling,
            init_std=init_std,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization)
        # DNN layer
        self.dense_len = dense_len
        dnn_input_size = self.embed_size * sparse_len + dense_len + input_size

        self.dnn = nn.ModuleList([nn.BatchNorm1d(dnn_input_size),
                                  nn.Linear(dnn_input_size, dnn_hidden_units[0]),
                                  nn.ReLU(),
                                  nn.Dropout(dnn_dropout),
                                  ])
        for i, hidden_layer in enumerate(dnn_hidden_units):
            if i == 0:
                continue
            else:
                self.dnn.extend([nn.Linear(dnn_hidden_units[i - 1], dnn_hidden_units[i]),
                                 nn.ReLU(),
                                 nn.Dropout(dnn_dropout)
                                 ])
        self.linear = nn.Linear(dnn_hidden_units[-1], 1)

    def forward(self, target_item, hist_item, neg_hist_item, hist_valid_lens, dense_feature, sparse_feature):

        target_item_emb = torch.flatten(self.embedding_features(target_item), start_dim=1)
        hist_item_emb = torch.flatten(self.embedding_features(hist_item), start_dim=2)
        neg_hist_item_emb = torch.flatten(self.embedding_features(neg_hist_item), start_dim=2)
        sparse_feature_emb = self.embedding_features(sparse_feature)
        sparse_feature_emb_fla = torch.flatten(sparse_feature_emb, start_dim=1)
        # [b, T, H],  [1]  (b<H)
        masked_interest, aux_loss = self.interest_extractor(hist_item_emb, hist_valid_lens,
                                                            neg_hist_item_emb)

        # [B, H]
        hist = self.interest_evolution(target_item_emb, masked_interest, hist_valid_lens)
        # [B, H2]
        deep_input_emb = torch.cat([hist, sparse_feature_emb_fla], dim=-1)
        if self.dense_len == 0:
            dense_feature = None
        if dense_feature is not None:
            dnn_input = torch.cat([deep_input_emb, dense_feature.reshape(-1, 1)], dim=-1)
        else:
            dnn_input = deep_input_emb

        for net in self.dnn:
            dnn_input = net(dnn_input)
        output = self.linear(dnn_input)
        y_pred = output
        return y_pred, aux_loss


if __name__ == "__main__":
    pass
