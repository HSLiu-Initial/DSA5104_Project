import math
import os
import pickle as pkl
import logging

import torch
from torch import nn
from tqdm import tqdm

# from ..utils.functions import get_dataset_summary_dict


def get_dataset_summary_dict(dataset_summary_file):
    if not os.path.exists(dataset_summary_file):
        logging.error('data summary file {} does not exists'.format(dataset_summary_file))
    with open(dataset_summary_file, 'rb') as f:
        summary_dict = pkl.load(f)
    return summary_dict


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    value = torch.tensor(value, dtype=X.dtype, device=X.device)
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=X.dtype,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


# @save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量,valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换,从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# @save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, d, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.attention_weights = None
        # self.dropout = nn.Dropout(dropout)
        self.att_mat = nn.Linear(d, d, bias=False)

    def forward(self, queries, keys, values, labels, valid_lens=None):
        """
        :param queries: (batch_size,查询的个数,d)
        :param keys: (batch_size,“键－值”对的个数,d)
        :param values: (batch_size,“键－值”对的个数,值的维度)
        :param valid_lens: (batch_size,)或者(batch_size,查询的个数)
        :return: values: (batch_size,值的维度)
        """
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        # print(queries,keys)
        scores = torch.bmm(self.att_mat(queries), keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # return torch.bmm(self.dropout(self.attention_weights), values), torch.bmm(self.dropout(self.attention_weights),
        #                                                                           labels)
        return torch.bmm(self.attention_weights, values), torch.bmm(self.attention_weights, labels)


class RIM(nn.Module):
    def __init__(self, dataset_summary_file, features, embed_size, dropout, hidden_layers):
        """
        :param dataset_summary_file: data/taobao/feateng_data/summary_dict.pkl文件
        :param features: 数据集中作为特征列数 Tmall：9
        :param embed_size: embedding维度
        :param hidden_layers:list 输出层的维度
        :param 是否使用embedding
        """
        super(RIM, self).__init__()
        self.features = features
        self.embed_size = embed_size
        self.dataset_summary = get_dataset_summary_dict(dataset_summary_file)

        self.embedding_features = nn.Embedding(num_embeddings=self.dataset_summary['feat_num'],
                                               embedding_dim=embed_size)

        self.embedding_labels = nn.Embedding(num_embeddings=2,
                                             embedding_dim=embed_size)
        self.aggre_atten = DotProductAttention(dropout=dropout, d=features * embed_size)
        self.output = nn.ModuleList([nn.Linear((2 * features + 1) * (embed_size + features), hidden_layers[0]),
                                     nn.BatchNorm1d(hidden_layers[0]),
                                     nn.ReLU()
                                     ])
        for i, hidden_layer in enumerate(hidden_layers[:-1]):
            if i == 0:
                continue
            else:
                self.output.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                                    nn.BatchNorm1d(hidden_layers[i]),
                                    nn.ReLU(),
                                    ])
        # self.out = nn.Sequential(nn.Linear(hidden_layers[-2] + embed_size, hidden_layers[-1]), nn.Sigmoid())
        self.out = nn.Linear(hidden_layers[-2] + embed_size, hidden_layers[-1])
        # self.out = nn.Sigmoid()

    # def forward(self, x, ret, ret_label, valid_lens, use_embed, prt_hint=False):
    def forward(self, x, use_embed, prt_hint=False):
        """
        :param x: [batch, features]或者是[batch, features,embed_size] 之前已经embedding好了
        :param ret: [batch, ret_size, features] 或者是[batch, ret_size, features,embed_size] 之前已经embedding好了
        :param ret_label: [batch, ret_size]
        :return:
        """
        # print(x, ret)
        if use_embed:
            # Embedding
            x_emb = self.embedding_features(x)  # [batch, features, embed_size]
            # ret_emb = self.embedding_features(ret)  # [batch, ret_size, features, embed_size]
            # ret_label_emb = self.embedding_labels(ret_label)  # [batch, ret_size, embed_size]
            batch = x_emb.shape[0]
        else:
            x_emb = x
            # ret_emb = ret
            # ret_label_emb = ret_label
            batch = x_emb.shape[0]
        # Aggregating
        # print(x,ret)
        x_emb_fla = torch.flatten(x_emb, start_dim=1).unsqueeze(1)  # [batch, 1, features*embed_size]
        # return x_emb_fla


        ret_emb = torch.flatten(ret_emb, start_dim=2)  # [batch, ret_size, features*embed_size]
        aggre_feat, aggre_lab = self.aggre_atten(queries=x_emb_fla, keys=ret_emb, values=ret_emb,  # 对检索样本进行聚合
                                                 labels=ret_label_emb,
                                                 valid_lens=valid_lens)  # [batch, 1, features*embed_size], [batch, 1, embed_size]
        aggre_res = torch.cat([aggre_feat.reshape(batch, self.features, -1), aggre_lab.reshape(batch, 1, -1)], dim=1)

        # Concatenating
        x_emb_fla_resh = x_emb_fla.reshape(batch, self.features, -1)  # [batch, features, embed_size]
        aggre_feat_resh = aggre_feat.reshape(batch, self.features, -1)  # [batch, features, embed_size]
        c_combine = torch.cat([x_emb_fla_resh, aggre_feat_resh, aggre_lab], dim=1)  # [batch, 2*features+1, embed_size]

        # 使用内积
        c_combine_inter = torch.bmm(c_combine, c_combine.permute(0, 2, 1))  # [batch, 2*features+1, 2*features+1]
        mask = torch.triu(torch.ones_like(c_combine_inter), diagonal=1)
        c_combine_inter_res = c_combine_inter[torch.nonzero(mask, as_tuple=True)]
        c_combine_inter_res = c_combine_inter_res.reshape(batch, -1)
        inp = torch.cat([x_emb_fla.squeeze(1), aggre_feat.squeeze(1), aggre_lab.squeeze(1), c_combine_inter_res],
                        dim=1)  # [batch,(2*features+1)*(embed_size+features)]
        # features*embed_size, features*embed_size, embed_size, features*(2*features+1)
        inp = torch.flatten(inp, start_dim=1)
        for net in self.output:
            inp = net(inp)
        inp = torch.cat([inp, aggre_lab.squeeze(1)], dim=1)
        inp = self.out(inp)
        if prt_hint:
            return inp, aggre_res
        else:
            return inp

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     logging.error('PLEASE INPUT [DATASET]')
    #     sys.exit(0)
    # dataset = sys.argv[1]
    #
    # # read config file
    # cnf = configparser.ConfigParser()
    # cnf.read('../../configs/config_datasets.ini', encoding='UTF-8')
    #
    # dic = get_dataset_summary_dict(cnf.get(dataset, 'summary_dict_file'))
    # print(dic)
    model = RIM(
        dataset_summary_file='./data/tmall/feateng_data/summary_dict.pkl',
        features=7, embed_size=16, dropout=0.3, hidden_layers=[200, 80, 1]
    )
    loaded_dict = torch.load('model.pth')
    model.load_state_dict(loaded_dict)

    for name, child in model.named_children():
        print(name, child)
    #
    x, ret, ret_label = torch.randint(0, 5, size=(2, 7)), torch.randint(0, 5, size=(2, 5, 7)), torch.randint(0, 1,
                                                                                                             size=(
                                                                                                                 2, 5))
    print(model(x, use_embed=True).shape)




