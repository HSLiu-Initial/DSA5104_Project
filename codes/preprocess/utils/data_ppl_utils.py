import pickle as pkl
import random
from operator import itemgetter

import numpy as np
from tqdm import tqdm


def dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('{} dumped'.format(filename))


def dump_lines(filename, lines):
    with open(filename, 'w') as f:
        f.writelines(lines)
    print('{} writelines completed'.format(filename))


def sampling(ori_file, sampled_file, rate):
    sampled_lines = []
    with open(ori_file) as f:
        for line in tqdm(f):
            r = random.randint(1, int(1 / rate))
            if r == 1:
                sampled_lines.append(line)
    dump_lines(sampled_file, sampled_lines)


def select_pos_list(input_list, pos_list):
    return np.array(input_list)[pos_list].tolist()


def select_pos_str(input_str, pos_list):
    return ','.join(np.array(input_str.split(','))[pos_list].tolist())


def get_feat(c_feature, cnf, dataset):
    temp = cnf.get(dataset, c_feature)
    if temp != 'None':
        if len(cnf.get(dataset, c_feature)) > 1:
            temp = list(map(int, cnf.get(dataset, c_feature).strip('\n').split(',')))
        else:
            temp = [cnf.getint(dataset, c_feature)]
    else:
        temp = None
    return temp


def get_keys_from_dict(dict, keys):
    if isinstance(keys, list):
        # need make sure keys in dict_key
        return list(itemgetter(*keys)(dict))
    else:
        return dict[str(keys)]


def get_value_from_numpyarray(array, pos, dict):
    return get_keys_from_dict(dict=dict, keys=array[pos])
