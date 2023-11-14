
import torch
import torch.nn as nn

from codes.utils.functions import get_dataset_summary_dict
from .utils.layers import FM, DNN



def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    value = torch.tensor(value, dtype=X.dtype, device=X.device)
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=X.dtype,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class DeepFM(nn.Module):

    def __init__(self,
                 dataset_summary_file,
                 feature_num,
                 embed_size,
                 item_feature_num, dnn_activation='relu', dnn_use_bn=True,
                 use_fm=True,
                 dnn_hidden_units=(200, 80), dnn_dropout=0.5, init_std=0.001):

        super(DeepFM, self).__init__()
        self.dataset_summary = get_dataset_summary_dict(dataset_summary_file)

        self.embedding_features = nn.Embedding(num_embeddings=self.dataset_summary['feat_num'],
                                               embedding_dim=embed_size)
        self.use_fm = use_fm
        self.use_dnn = len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            dnn_input_size = (feature_num + item_feature_num) * embed_size
            self.dnn = DNN(dnn_input_size, dnn_hidden_units,
                           activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1)
        self.out = nn.Sigmoid()

    def forward(self, X, hist_item, hist_valid_lens, use_emb=False):

        logit = 0
        if use_emb:
            X = self.embedding_features(X)
            hist_item = self.embedding_features(hist_item)  # [B,T,H,embedding_size]

        hist_item_mask = sequence_mask(hist_item, valid_len=hist_valid_lens, value=0)  # [B,T,H,embedding_size]
        ones = torch.ones_like(hist_valid_lens)
        hist_valid_lens = torch.where(hist_valid_lens != 0, hist_valid_lens, ones)
        hist_item_mask_sum = torch.sum(hist_item_mask, dim=1)  # [B,H,embedding_size]
        hist_valid_lens_trans = hist_valid_lens.unsqueeze(-1).unsqueeze(-1).repeat(1, hist_item_mask_sum.shape[1],
                                                                                   hist_item_mask_sum.shape[2])
        hist_item_mean = (hist_item_mask_sum / hist_valid_lens_trans)  # [B,H,embedding_size]

        X = torch.cat([X, hist_item_mean], dim=1)
        if self.use_fm:
            fm_input = X
            logit += self.fm(fm_input)  # (batch_size, select_size**2,1)

        if self.use_dnn:
            dnn_input = torch.flatten(X, start_dim=1)  # [batch,query_size*embed_size]
            dnn_output = self.dnn(dnn_input, input_4d=False)
            dnn_logit = self.dnn_linear(dnn_output)  # [batch,1]
            logit += dnn_logit

        # y_pred = self.out(logit)
        y_pred = logit

        return y_pred
