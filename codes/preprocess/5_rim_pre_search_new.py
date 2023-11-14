import configparser
import logging
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_ppl_utils import dump_lines
from utils.es_reader import queryGen, ESReader

logging.basicConfig(level=logging.INFO)


def pre_search_rim(query_generator, es_reader, search_res_col_file, sequential=False):

    search_res_col_lines = []
    search_res_label_lines = []
    query_fn = es_reader.query_rim1 if sequential else es_reader.query_rim_ac

    for batch in tqdm(query_generator, total=query_generator.total_step, dynamic_ncols=True):
        q_batch, sync_id_batch = batch  #
        res_lineno_batch = query_fn(q_batch, sync_id_batch)

        search_res_col_lines += [(','.join(res) + '\n') for res in res_lineno_batch]  #

    dump_lines(search_res_col_file, search_res_col_lines)





def merge_files(base_folder, format, update_target=False):
    if update_target or not os.path.exists(os.path.join(base_folder, 'target_train' + format)):
        print('Writing train target file...')
        X = np.load(os.path.join(base_folder, 'train_input_all.npy'))
        pd.DataFrame(X, index=None).to_csv(
            os.path.join(base_folder, 'target_train' + format), sep=',', header=False, index=False)
    else:
        print('Skip train target file writing...')

    if update_target or not os.path.exists(os.path.join(base_folder, 'target_test' + format)):
        print('Writing test target file...')
        X = np.load(os.path.join(base_folder, 'test_input_part_0.npy'))
        pd.DataFrame(X, index=None).to_csv(
            os.path.join(base_folder, 'target_test' + format), sep=',', header=False, index=False)
    else:
        print('Skip test target file writing...')


available_datasets = ['avazu', 'criteo', 'taobao', 'tmall', 'alipay']
sequential_datasets = ['taobao', 'tmall', 'alipay']

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('PLEASE INPUT [DATASET] [BATCH_SIZE] [RETRIEVE_SIZE] [MODE]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = int(sys.argv[2])
    # mode = sys.argv[4]

    # read config file
    cnf = configparser.ConfigParser()
    file_path = os.path.abspath(__file__)
    root_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    cnf.read(os.path.join(root_dir_path, 'configs/config_datasets.ini'), encoding='UTF-8')
    size = cnf.getint(dataset, 'ret_size')
    # query generator
    query_generator_train = queryGen(os.path.join(root_dir_path, cnf.get(dataset, 'target_train_sample_file')),
                                     batch_size,
                                     cnf.getint(dataset, 'sync_c_pos'),
                                     cnf.get(dataset, 'query_c_pos'))
    query_generator_test = queryGen(os.path.join(root_dir_path, cnf.get(dataset, 'target_test_sample_file')),
                                    batch_size,
                                    cnf.getint(dataset, 'sync_c_pos'),
                                    cnf.get(dataset, 'query_c_pos'))
    es_reader = ESReader(dataset, size)  # size

    logging.info('target train pre searching...')
    pre_search_rim(query_generator_train,
                   es_reader,
                   os.path.join(root_dir_path,
                                f'data/{dataset}/feateng_data/ret_res/search_res_col_train_{size}_sample.txt'),
                   dataset in sequential_datasets)

    logging.info('target test pre searching...')
    pre_search_rim(query_generator_test,
                   es_reader,
                   os.path.join(root_dir_path,
                                f'data/{dataset}/feateng_data/ret_res/search_res_col_test_{size}_sample.txt'),
                   dataset in sequential_datasets)