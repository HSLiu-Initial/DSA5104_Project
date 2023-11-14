# -*- coding:utf-8 -*-

import configparser
import logging
import os
import pickle as pkl
import sys

from tqdm import tqdm


def dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('{} dumped'.format(filename))


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error('PLEASE INPUT [DATASET] [ret_size]')
        sys.exit(0)
    dataset = sys.argv[1]
    ret_size = int(sys.argv[2])
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

    # target_train_file = os.path.join(root_dir_path, cnf.get(dataset, 'target_train_sample_file'))
    # target_test_file = os.path.join(root_dir_path, cnf.get(dataset, 'target_test_sample_file'))
    sync_seq_dict_file = os.path.join(root_dir_path, cnf.get(dataset, 'sync_seq_dict_file'))
    neg_sync_seq_dict_file = os.path.join(root_dir_path, cnf.get(dataset, 'neg_sync_seq_dict_file'))
    hist_item_preprocessed_file = os.path.join(root_dir_path,
                                               cnf.get(dataset, 'hist_item_preprocessed_file')[::-1].replace(".",
                                                                                                             f"_{ret_size}."[
                                                                                                             ::-1], 1)[
                                               ::-1])
    neg_hist_item_preprocessed_file = os.path.join(root_dir_path,
                                                   cnf.get(dataset, 'neg_hist_item_preprocessed_file')[::-1].replace(
                                                       ".", f"_{ret_size}."[::-1], 1)[::-1])
    hist_valid_lens_preprocessed_file = os.path.join(root_dir_path,
                                                     cnf.get(dataset, 'hist_valid_lens_preprocessed_file')[
                                                     ::-1].replace(".", f"_{ret_size}."[::-1], 1)[::-1])

    with open(
            sync_seq_dict_file,
            'rb') as f:
        hist_item = pkl.load(f)
    with open(
            neg_sync_seq_dict_file,
            'rb') as f:
        neg_hist_item = pkl.load(f)
    hist_item_preprocessed = {}
    neg_hist_item_preprocessed = {}
    hist_valid_lens_preprocessed = {}
    for key, value in tqdm(hist_item.items()):
        if len(hist_item[key]) == 0:
            continue
        else:
            length = len(hist_item[key])
            if length >= ret_size:
                hist_item_temp = list(
                    map(lambda x: list(map(int, x.split(','))),
                        list(map(lambda x: x[0], hist_item[key][-ret_size:]))))
                neg_hist_item_temp = list(map(lambda x: list(map(int, x.split(','))),
                                              list(map(lambda x: x[0], neg_hist_item[key][-ret_size:]))))
                hist_valid_lens_temp = ret_size
            else:
                hist_item_temp = list(
                    map(lambda x: list(map(int, x.split(','))),
                        list(map(lambda x: x[0], hist_item[key])))) + [[0] * len(sampling_c_pos_list)] * (
                                         ret_size - length)
                neg_hist_item_temp = list(map(lambda x: list(map(int, x.split(','))),
                                              list(map(lambda x: x[0], neg_hist_item[key])))) + [
                                         [0] * len(sampling_c_pos_list)] * (
                                             ret_size - length)
                hist_valid_lens_temp = length
            hist_item_preprocessed[key] = hist_item_temp
            neg_hist_item_preprocessed[key] = neg_hist_item_temp
            hist_valid_lens_preprocessed[key] = hist_valid_lens_temp
    dump_pkl(hist_item_preprocessed_file, hist_item_preprocessed)
    dump_pkl(neg_hist_item_preprocessed_file, neg_hist_item_preprocessed)
    dump_pkl(hist_valid_lens_preprocessed_file, hist_valid_lens_preprocessed)
