# -*- coding:utf-8 -*-

import configparser
import logging
import os
import pickle as pkl
import sys

import numpy as np
from tqdm import tqdm


def dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('{} dumped'.format(filename))



logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]
    # read config file
    cnf = configparser.ConfigParser()
    file_path = os.path.abspath(__file__)
    root_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    cnf.read(os.path.join(root_dir_path, 'configs/config_datasets.ini'), encoding='UTF-8')
    # temp = pd.read_csv(cnf.get(dataset, 'search_pool_file'), header=None)
    # res = temp.sort_values(by=[0, 4], ascending=[True, True])
    # res.to_csv(cnf.get(dataset, 'search_pool_sorted_file'), header=None, index=None)
    sync_c_pos = cnf.getint(dataset, 'sync_c_pos')  # 0
    sampling_c_pos_list = list(map(int, cnf.get(dataset, 'sampling_c_pos_list').split(',')))  # 1 2
    timestamp_pos = cnf.getint(dataset, 'timestamp_pos')  # 4
    neg_sync_seq_dict_file = cnf.get(dataset, 'neg_sync_seq_dict_file')
    # with open("sync_seq_dict_file", 'rb') as pos:
    #     pos_dic = pkl.load(pos)
    neg_sync_seq_dict = {}
    num_file = sum([1 for i in open(os.path.join(root_dir_path, cnf.get(dataset, 'search_pool_file')))])
    with open(os.path.join(root_dir_path, cnf.get(dataset, 'search_pool_file'))) as search_pool_lines:
        for line in tqdm(search_pool_lines,
                         total=num_file):  #
            line_split = line[:-1].split(',')
            if line_split[-1] == '0':
                sync_id = line_split[sync_c_pos]  # userid
                if sync_id in neg_sync_seq_dict:  #
                    seq_part = ','.join(np.array(line_split)[sampling_c_pos_list].tolist())
                    neg_sync_seq_dict[sync_id].append((seq_part, int(float(line_split[timestamp_pos]))))
                else:
                    neg_sync_seq_dict[sync_id] = []
                    seq_part = ','.join(np.array(line_split)[sampling_c_pos_list].tolist())
                    neg_sync_seq_dict[sync_id].append((seq_part, int(float(line_split[timestamp_pos]))))

    logging.info('sync_seq_dict sorting...')
    for sync_id in tqdm(neg_sync_seq_dict):
        neg_sync_seq_dict[sync_id].sort(key=lambda x: x[1])

    dump_pkl(os.path.join(root_dir_path, neg_sync_seq_dict_file), neg_sync_seq_dict)
