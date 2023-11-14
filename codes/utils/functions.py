# -*- coding:utf-8 -*-

import argparse
import copy
import json
import logging
import os
import pickle as pkl
import pprint
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter


class ConfigParser(object):
    def __init__(self, options):
        for i, j in options.items():
            if isinstance(j, dict):
                for k, v in j.items():
                    setattr(self, k, v)
            else:
                setattr(self, i, j)
        self._device()

    def update(self, args):
        for args_k in args.__dict__:
            # assert hasattr(self, args_k) or args_k == 'config', "Please check your setting"
            if getattr(args, args_k) is not None:
                setattr(self, args_k, getattr(args, args_k))
        # if self.trade_mode == 'D':
        #     self.trade_len = 1
        # elif self.trade_mode == 'W':
        #     self.trade_len = 5
        # elif self.trade_mode == 'M':
        #     self.trade_len = 21
        # else:
        #     raise ValueError
        self._device()

    def _device(self):
        if self.gpu != -1 and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def save(self, save_dir):
        dic = self.__dict__
        dic['device'] = 'cuda' if dic['device'] == torch.device('cuda') else 'cpu'
        js = json.dumps(dic)
        with open(save_dir, 'w') as f:
            f.write(js)


def setup_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_dataset_summary_dict(dataset_summary_file):
    if not os.path.exists(dataset_summary_file):
        logging.error('data summary file {} does not exists'.format(dataset_summary_file))
    with open(dataset_summary_file, 'rb') as f:
        summary_dict = pkl.load(f)
    return summary_dict


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def update_args_(args, params):
    """updates args in-place"""
    dargs = vars(args)
    dargs.update(params)


def init_method(PREFIX, args):

    img_dir = os.path.join(PREFIX, 'img_file')
    save_dir = os.path.join(PREFIX, 'log_file')
    model_save_dir = os.path.join(PREFIX, 'model_file')
    results_save_dir = os.path.join(PREFIX, 'results_file')
    train_dir = os.path.join(results_save_dir, "train")
    eval_dir = os.path.join(results_save_dir, "evale")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.isdir(results_save_dir):
        os.makedirs(results_save_dir)
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    try:
        hyper = copy.deepcopy(args.__dict__)
        pprint.pprint(hyper)
        hyper['device'] = 'cuda'
        json_str = json.dumps(hyper, indent=4)
        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)
    except:
        print("xxx args")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('WARNING')
    fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    train_writer = SummaryWriter(log_dir=train_dir)
    eval_writer = SummaryWriter(log_dir=eval_dir)
    return model_save_dir, train_writer, eval_writer


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
