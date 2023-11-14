import configparser
import pickle as pkl
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset_DIEN(Dataset):
    def __init__(self, target_data, hist_item_file, neg_hist_item_file, hist_valid_lens, dense_feature_c_pos,
                 sparse_feature_c_pos, target_c_pos, c_user_id, ret_size=20
                 ):
        self.ret_size = ret_size
        self.c_user_id = c_user_id
        self.lines = pd.read_csv(target_data, header=None, sep=',', encoding='latin-1').astype(int).values
        with open(
                hist_item_file,
                'rb') as f:
            self.hist_item = pkl.load(f)
        with open(
                neg_hist_item_file,
                'rb') as f:
            self.neg_hist_item = pkl.load(f)
        with open(
                hist_valid_lens,
                'rb') as f:
            self.hist_valid_lens = pkl.load(f)

        self.label = self.lines[:, -1]
        self.len = self.lines.shape[0]

        self.target_c_pos = target_c_pos
        self.dense_feature_c_pos = dense_feature_c_pos
        self.sparse_feature_c_pos = sparse_feature_c_pos


    def __getitem__(self, index):
        """
        :return target_item: [batch_size, H]
        :return hist_item: [B, T, H]
        :return neg_hist_item: [B, T, H]
        :return hist_valid_lens: [B,]
        :return dense_feature: [batch_size, dense_feature]
        :return sparse_feature: [batch_size, sparse_feature]
        """
        # flag = False
        # for value in self.hist_item.values():
        #     if len(value[0]) == len(self.target_c_pos):
        #         flag = True
        #         break
        y = self.label[index]
        user_id = self.lines[index, self.c_user_id]
        if self.dense_feature_c_pos is not None:
            dense_feature = self.lines[index, self.dense_feature_c_pos]
        else:
            dense_feature = -1
        sparse_feature = self.lines[index, self.sparse_feature_c_pos]
        target_item = self.lines[index, self.target_c_pos]
        if str(user_id) in self.hist_item:
            # if flag is True:
            #     hist_item = self.hist_item[str(user_id)]
            #     neg_hist_item = self.neg_hist_item[str(user_id)]
            #     hist_valid_lens = self.hist_valid_lens[str(user_id)]
            # else:
            hist_item = list(map(lambda x: x[:len(self.target_c_pos)], self.hist_item[str(user_id)]))
            neg_hist_item = list(map(lambda x: x[:len(self.target_c_pos)], self.neg_hist_item[str(user_id)]))
            hist_valid_lens = self.hist_valid_lens[str(user_id)]
        else:
            hist_item = [[0] * len(self.target_c_pos)] * self.ret_size
            neg_hist_item = [[0] * len(self.target_c_pos)] * self.ret_size
            hist_valid_lens = 0
        dic = {'target_item': torch.tensor(target_item),
               'hist_item': torch.tensor(hist_item),
               'neg_hist_item': torch.tensor(neg_hist_item),
               'hist_valid_lens': torch.tensor(hist_valid_lens), 'dense_feature': torch.tensor(dense_feature),
               'sparse_feature': torch.tensor(sparse_feature),
               'y': y}
        return dic

    def __len__(self):
        return self.len


if __name__ == "__main__":
    pass
