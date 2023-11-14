import configparser
import pandas as pd
import sys
from torch.utils.data import Dataset, DataLoader

class Dataset_RIM(Dataset):
    def __init__(self, target_data, neighbor_idxs, retrieve_pool, query_c_pos, ret_size=None):
        self.ret_size = ret_size
        self.lines = pd.read_csv(target_data, header=None, sep=',', encoding='latin-1').astype(int).values
        self.label = self.lines[:, -1]
        self.len = self.lines.shape[0]
        with open(neighbor_idxs, 'r') as f:
            self.knn_idxs = f.readlines()
        self.retrieval_pool = pd.read_csv(retrieve_pool, header=None, sep=',', encoding='latin-1').astype(
            int).values
        self.query_c_pos = list(map(int, query_c_pos.split(',')))

    def __getitem__(self, index):
        # idxs, neighbors, user, movie, genres, gender, age, occupation, label = items
        neighbor_idxs = self.knn_idxs[index]
        neighbors = list(map(int, neighbor_idxs.split(',')))
        if self.ret_size != None:
            # raise NotImplementedError
            neighbors += [neighbors[0]] * (
                    self.ret_size - len(neighbors))
        y = self.label[index]
        x = self.lines[index, self.query_c_pos]  # [features, ]
        ret = self.retrieval_pool[neighbors][:, self.query_c_pos]  # [ret_size, features+1]
        ret_label = self.retrieval_pool[neighbors][:, -1]  # [batch]
        valid_lens = len(set(neighbors))
        dic = {'x': x, 'y': y, 'ret': ret, 'ret_label': ret_label, 'valid_lens': valid_lens}
        return dic  # batch后：x: [batch,features], y: [batch,], ret: [batch, ret_size, features], ret_label: [batch,]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    pass
