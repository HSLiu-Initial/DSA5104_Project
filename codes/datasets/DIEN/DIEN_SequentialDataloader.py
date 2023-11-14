import configparser
import os
import sys
import torch
from torch.utils.data import DataLoader

from codes.datasets.DIEN.DIEN_SequentialDatasets import Dataset_DIEN

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def load_data(dataset, batch_size, args, ret_size=739, num_workers=0, path='./data', use_ddp=False):
    path_2 = os.path.dirname(path)  # homepath
    # path_2 = "../../.."
    # print(path_2)
    # read config file
    cnf = configparser.ConfigParser()
    cnf.read(os.path.join(path_2, 'configs/config_datasets.ini'),
             encoding='UTF-8')
    # data/dataset/feateng_data
    # path = '../../../data'
    prefix = os.path.join(path, dataset, 'feateng_data')
    target_train = os.path.join(prefix, 'target', 'target_train_sample.csv')
    target_test = os.path.join(prefix, 'target', 'target_test_sample.csv')
    hist_item_file = os.path.join(path_2, cnf.get(dataset, 'hist_item_preprocessed_file')[::-1].replace(".",
                                                                                                        f"_{ret_size}."[
                                                                                                        ::-1], 1)[
                                          ::-1])
    neg_hist_item_file = os.path.join(path_2, cnf.get(dataset, 'neg_hist_item_preprocessed_file')[::-1].replace(".",
                                                                                                                f"_{ret_size}."[
                                                                                                                ::-1],
                                                                                                                1)[
                                              ::-1])
    hist_valid_lens = os.path.join(path_2, cnf.get(dataset, 'hist_valid_lens_preprocessed_file')[::-1].replace(".",
                                                                                                               f"_{ret_size}."[
                                                                                                               ::-1],
                                                                                                               1)[
                                           ::-1])

    if dataset == 'tmall':
        if args.only_idcat:
            sparse_feature_c_pos = [0, 1, 2, 3, 4, 7, 8]
            target_c_pos = [1, 2]
            c_user_id = 0
        else:
            sparse_feature_c_pos = [0, 1, 2, 3, 4, 7, 8]
            target_c_pos = [1, 2, 3, 4]
            c_user_id = 0
    elif dataset == 'taobao':
        sparse_feature_c_pos = [0, 1, 2]
        target_c_pos = [1, 2]
        c_user_id = 0
    elif dataset == 'alipay':
        sparse_feature_c_pos = [0, 1, 2, 3]
        target_c_pos = [1, 2, 3]
        c_user_id = 0
    else:
        raise NotImplementedError
    train_dataset = Dataset_DIEN(target_data=target_train, hist_item_file=hist_item_file,
                                 neg_hist_item_file=neg_hist_item_file, hist_valid_lens=hist_valid_lens,
                                 dense_feature_c_pos=None,
                                 sparse_feature_c_pos=sparse_feature_c_pos, target_c_pos=target_c_pos,
                                 c_user_id=c_user_id,
                                 ret_size=ret_size
                                 )
    test_dataset = Dataset_DIEN(target_data=target_test, hist_item_file=hist_item_file,
                                neg_hist_item_file=neg_hist_item_file, hist_valid_lens=hist_valid_lens,
                                dense_feature_c_pos=None,
                                sparse_feature_c_pos=sparse_feature_c_pos, target_c_pos=target_c_pos,
                                c_user_id=c_user_id,
                                ret_size=ret_size
                                )

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                        rank=args.rank)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   sampler=train_sampler, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.world_size,
                                                                       rank=args.rank)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  sampler=test_sampler, drop_last=True)
    else:
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":
    pass