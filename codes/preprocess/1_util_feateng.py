# sys.path.append("..")
import configparser
import logging
import os
import sys

import pandas as pd

from utils.data_ppl_utils import *

logging.basicConfig(level=logging.INFO)

random.seed(1111)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def gen_remap_dicts(joined_tabular_file,
                    used_cnum,
                    remap_c_pos_list,
                    remap_dicts_file,
                    sep,
                    header):


    logging.info('remapping dicts begin...')
    # comes from the config file
    remap_c_pos_list = list(map(int, remap_c_pos_list.split(',')))  #
    assert used_cnum == len(remap_c_pos_list)

    # sets and remap dicts for each column that needs to be remapped
    c_sets = [set() for _ in range(used_cnum)]
    remap_dicts = [{} for _ in range(used_cnum)]

    # get all the unique feature values
    num_file = sum([1 for i in open(joined_tabular_file)])
    with open(joined_tabular_file) as f:
        if header == True:
            f.readline()  #
        for line in tqdm(f, total=num_file):
            line_split = line[:-1].split(sep=sep)
            for i, c_pos in enumerate(remap_c_pos_list):  #
                c_sets[i].add(line_split[c_pos])

    # generate remap dicts
    remap_id = 1
    for i, c_set in enumerate(c_sets):  #
        for c in c_set:  #
            remap_dicts[i][c] = str(remap_id)
            remap_id += 1
    logging.info('total feature number is: {}'.format(remap_id))

    dump_pkl(remap_dicts_file, remap_dicts)  #


def remap(joined_tabular_file,
          remap_dicts_file,
          remap_c_pos_list,
          sampling_c_pos_list,
          remapped_tabular_file,
          sampling_collection_file,
          dataset_summary_file,
          sep,
          header):

    remap_c_pos_list = list(map(int, remap_c_pos_list.split(',')))
    sampling_c_pos_list = list(map(int, sampling_c_pos_list.split(',')))

    with open(remap_dicts_file, 'rb') as f:  #
        remap_dicts = pkl.load(f)
        logging.info('remap_dicts have been loaded')

    remapped_tabular_lines = []
    sampling_collection_set = set()
    num_file = sum([1 for i in open(joined_tabular_file)])
    with open(joined_tabular_file) as f:  #
        if header == True:
            f.readline()
        for line in tqdm(f, total=num_file):
            line_split = line[:-1].split(sep=sep)
            for i, c_pos in enumerate(remap_c_pos_list):
                line_split[c_pos] = remap_dicts[i][line_split[c_pos]]  #
            remapped_tabular_lines.append(','.join(line_split) + '\n')  #

            sampling_c_list = list(np.array(line_split)[sampling_c_pos_list])  #
            sampling_collection_set.add(','.join(sampling_c_list))

    sampling_collection_list = list(sampling_collection_set)

    dump_lines(remapped_tabular_file, remapped_tabular_lines)
    dump_pkl(sampling_collection_file, sampling_collection_list)

    logging.info('remapped and sampling collection files dumped')
    logging.info('generating dataset summary file...')

    # generate summary file: columns info
    summary_dict = {}
    total_feat_num = 0
    for i, c_pos in enumerate(remap_c_pos_list):
        summary_dict['C{}'.format(c_pos)] = len(remap_dicts[i])  #
        total_feat_num += len(remap_dicts[i])
        logging.info('the number of column C{}\'s unique values(features) is {}'.format(c_pos, len(remap_dicts[i])))
    summary_dict['feat_num'] = total_feat_num + 1  #
    logging.info('total feature number is {}'.format(total_feat_num + 1))

    dump_pkl(dataset_summary_file, summary_dict)


def keep_pv(raw_data_file, joined_tabular_file, c_interaction_feature, name_click, header):

    header_r = None if header is False else 0
    tab = pd.read_csv(raw_data_file, header=header_r)
    name_click = int(name_click) if is_number(name_click) else name_click
    c_interaction_feature = int(c_interaction_feature) if is_number(c_interaction_feature) else c_interaction_feature
    tab_clean = tab[tab[c_interaction_feature] == name_click]
    # del tab_clean[3]
    if header_r == 0:
        tab_clean.to_csv(joined_tabular_file, index=False)
    else:
        tab_clean.to_csv(joined_tabular_file, header=header_r, index=False)


if __name__ == "__main__":
    """
    """
    if len(sys.argv) < 2:
        logging.error('PLEASE INPUT [DATASET]')
        sys.exit(0)
    dataset = sys.argv[1]
    # read config file
    cnf = configparser.ConfigParser()
    file_path = os.path.abspath(__file__)
    root_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

    cnf.read(os.path.join(root_dir_path, 'configs/config_datasets.ini'), encoding='UTF-8')

    # p1 = Path("../../data") / dataset / "feateng_data/temp"
    p1 = os.path.join(root_dir_path, "data", dataset, "feateng_data", "temp")
    p2 = os.path.join(root_dir_path, "data", dataset, "feateng_data", "target")
    p3 = os.path.join(root_dir_path, "data", dataset, "feateng_data", "ret_res")
    if not os.path.isdir(p1):
        os.makedirs(p1)
    if not os.path.isdir(p2):
        os.makedirs(p2)
    if not os.path.isdir(p3):
        os.makedirs(p3)
    if dataset == "tmall" or dataset == "taobao" or dataset == "alipay":
        #
        keep_pv(raw_data_file=os.path.join(root_dir_path, cnf.get(dataset, 'raw_data_file')),
                joined_tabular_file=os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_file')),
                c_interaction_feature=cnf.get(dataset, 'c_interaction_feature'),
                name_click=cnf.get(dataset, 'name_click'),
                header=cnf.getboolean(dataset, 'header'))
    if dataset == "tmall" or dataset == "taobao" or dataset == "alipay":
        # call functions
        gen_remap_dicts(os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_file')),
                        cnf.getint(dataset, 'used_cnum'),
                        cnf.get(dataset, 'remap_c_pos_list'),
                        os.path.join(root_dir_path, cnf.get(dataset, 'remap_dicts_file')),
                        sep=',',
                        header=cnf.getboolean(dataset, 'header'))

        remap(os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'remap_dicts_file')),
              cnf.get(dataset, 'remap_c_pos_list'),
              cnf.get(dataset, 'sampling_c_pos_list'),
              os.path.join(root_dir_path, cnf.get(dataset, 'remapped_tabular_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'sampling_collection_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'summary_dict_file')),
              sep=',',
              header=cnf.getboolean(dataset, 'header'))
    elif dataset == "avazu":
        # call functions
        gen_remap_dicts(os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_file')),
                        cnf.getint(dataset, 'used_cnum'),
                        cnf.get(dataset, 'remap_c_pos_list'),
                        os.path.join(root_dir_path, cnf.get(dataset, 'remap_dicts_file')),
                        sep=',',
                        header=cnf.getboolean(dataset, 'header'))

        remap(os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_train_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'remap_dicts_file')),
              cnf.get(dataset, 'remap_c_pos_list'),
              cnf.get(dataset, 'sampling_c_pos_list'),
              os.path.join(root_dir_path, cnf.get(dataset, 'remapped_tabular_train_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'sampling_collection_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'summary_dict_file')),
              sep=',',
              header=cnf.getboolean(dataset, 'header'))

        remap(os.path.join(root_dir_path, cnf.get(dataset, 'joined_tabular_test_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'remap_dicts_file')),
              cnf.get(dataset, 'remap_c_pos_list'),
              cnf.get(dataset, 'sampling_c_pos_list'),
              os.path.join(root_dir_path, cnf.get(dataset, 'remapped_tabular_test_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'sampling_collection_file')),
              os.path.join(root_dir_path, cnf.get(dataset, 'summary_dict_file')),
              sep=',',
              header=cnf.getboolean(dataset, 'header'))
