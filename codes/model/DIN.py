# -*- coding:utf-8 -*-

import torch

from codes.model.RIM import get_dataset_summary_dict
from codes.model.utils.layers import AttentionSequencePoolingLayer
from torch import nn


class DIN(nn.Module):

    def __init__(self, dataset_summary_file, embed_size,
                 dnn_hidden_units=(200, 80), att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=True,
                 dnn_dropout=0,
                 hist_itemfeature_dict_embedsize=None, sparse_len=None, dense_len=None):
        super(DIN, self).__init__()
        self.embed_size = embed_size
        self.dataset_summary = get_dataset_summary_dict(dataset_summary_file)
        self.embedding_features = nn.Embedding(num_embeddings=self.dataset_summary['feat_num'],
                                               embedding_dim=self.embed_size)

        input_size = sum([i for i in hist_itemfeature_dict_embedsize.values()])
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=input_size,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)
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

    def forward(self, target_item, hist_item, hist_valid_lens, sparse_feature):
        query_emb = torch.flatten(self.embedding_features(target_item), start_dim=1).unsqueeze(1)
        keys_emb = torch.flatten(self.embedding_features(hist_item), start_dim=2)
        sparse_feature_emb = torch.flatten(self.embedding_features(sparse_feature), start_dim=1).unsqueeze(1)
        keys_length = hist_valid_lens
        hist = self.attention(query_emb, keys_emb, keys_length)  # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((sparse_feature_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        for net in self.dnn:
            deep_input_emb = net(deep_input_emb)
        output = self.linear(deep_input_emb)
        # y_pred = self.out(output)
        y_pred = output

        return y_pred


if __name__ == '__main__':
    pass
