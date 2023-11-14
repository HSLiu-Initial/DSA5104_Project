import configparser
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from RIM_SequentialDatasets import Dataset_RIM


def load_data(dataset, batch_size, args, ret_size=10, num_workers=0, path='./data', use_ddp=False):
    path_2 = os.path.dirname(path)  # homepath
    print("path2:", path_2)
    print("path:", path)
    # read config file
    cnf = configparser.ConfigParser()
    cnf.read(os.path.join(path_2, 'config_datasets.ini'),
             encoding='UTF-8')
    # data/dataset/feateng_data
    prefix = os.path.join(path, dataset, 'feateng_data')
    target_train_full = os.path.join(prefix, 'target', 'target_train.csv')
    target_test_full = os.path.join(prefix, 'target', 'target_test.csv')
    target_train = os.path.join(prefix, 'target', 'target_train_sample.csv')
    target_test = os.path.join(prefix, 'target', 'target_test_sample.csv')

    train_knn_neighbors = os.path.join(prefix, 'ret_res', f'search_res_col_train_{ret_size}_sample.txt')
    test_knn_neighbors = os.path.join(prefix, 'ret_res', f'search_res_col_test_{ret_size}_sample.txt')

    retrieval_pool = os.path.join(prefix, 'target', 'search_pool.csv')

    if dataset == 'tmall' or dataset == 'taobao' or dataset == 'alipay':
        train_dataset = Dataset_RIM(target_train, train_knn_neighbors, retrieval_pool,
                                    cnf.get(dataset, 'query_c_pos_actual'), ret_size=ret_size)
        test_dataset = Dataset_RIM(target_test, test_knn_neighbors, retrieval_pool,  # 只使用remap_c_pos_list中涉及到的特征
                                   cnf.get(dataset, 'query_c_pos_actual'), ret_size=ret_size)
        # ret_dataset = Dataset_RIM(retrieval_pool, test_knn_neighbors, retrieval_pool,
        #                            cnf.get(dataset, 'query_c_pos_actual'), ret_size=ret_size)
    else:
        raise NotImplementedError

    if use_ddp:  # 如果使用DDP进行分布式训练
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
            dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # ret_loader = DataLoader(
        #     dataset=ret_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def load_data_testspeed(dataset, batch_size, args, ret_size=10, num_workers=0, path='./data', use_ddp=False):
    path_2 = os.path.dirname(path)  # homepath
    print("path2:", path_2)
    print("path:", path)
    # read config file
    cnf = configparser.ConfigParser()
    cnf.read(os.path.join(path_2, 'configs/config_datasets.ini'),
             encoding='UTF-8')
    # data/dataset/feateng_data
    prefix = os.path.join(path, dataset, 'feateng_data')
    target_test = os.path.join(prefix, 'target', 'target_test_sample_testspeed.csv')
    test_knn_neighbors = os.path.join(prefix, 'ret_res', f'search_res_col_test_{ret_size}_sample.txt')

    retrieval_pool = os.path.join(prefix, 'target', 'search_pool.csv')

    if dataset == 'tmall' or dataset == 'taobao' or dataset == 'alipay':
        test_dataset = Dataset_RIM(target_test, test_knn_neighbors, retrieval_pool,  # 只使用remap_c_pos_list中涉及到的特征
                                   cnf.get(dataset, 'query_c_pos_actual'), ret_size=ret_size)
    else:
        raise NotImplementedError

    test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_loader

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print('PLEASE INPUT [DATASET]')
    #     sys.exit(0)
    # dataset = sys.argv[1]
    batch_size = 100
    train_loader, test_loader = load_data(
        'tmall', 100, [], 10, 0
    )
    for i, data in enumerate(train_loader):
        # t = time.time()
        x = data['x']
        y = data['y']
        ret = data['ret']
        ret_label = data['ret_label']
        print(x.shape)
        print(y.shape)
        print(ret.shape)
        print(ret_label.shape)
        print(type(x))
        break