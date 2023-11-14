import argparse
import configparser
import json
import logging
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from torch.cuda.amp import GradScaler

from codes.datasets.DIEN import DIEN_SequentialDataloader
from codes.model.DIEN import DIEN
from codes.model.DIN import DIN
from codes.train_evaluate.train_evaluate_dien import test_dien, train_validate_dien
from codes.train_evaluate.train_evaluate_din import train_validate_din, test_din
from codes.utils.functions import ConfigParser, setup_seed, init_method





def main(args):
    # ===========================================================
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if args.resume is True and args.prefix == "auto define":
        raise Exception("no prefix")
    try:
        logging.info(args.rank)
        if args.use_ddp is False:
            raise Exception(f"use_ddp is {args.use_ddp}")
    except:
        if args.use_ddp is True:
            raise Exception(f"use_ddp is {args.use_ddp},no ddp")
    # if args.
    if args.use_ddp:
        # dist.init_process_group(backend='ncll', init_method='env://')
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                world_size=args.world_size)
        if args.rank == 0:
            logging.info(f"ddp；world size:{args.world_size};batch size:{args.batch_size};lr:{args.lr}")

    file_path = os.path.abspath(__file__)
    file_dir_path = os.path.dirname(file_path)

    cnf = configparser.ConfigParser()
    cnf.read(os.path.join(file_dir_path,
                          'configs/config_datasets.ini'),
             encoding='UTF-8')
    if args.seed != -1:
        try:
            setup_seed(args.seed + args.rank)
        except:
            setup_seed(args.seed)
    start_time_date = datetime.now().strftime('%m%d')
    start_time_time = datetime.now().strftime('%H_%M_%S')

    # output/<model>/<dataset>/<date>/<time>
    # PREFIX = Path(args.out_path) / 'run' / args.model / args.dataset / start_time_date / start_time_time
    if args.prefix == 'auto define':
        PREFIX = os.path.join(file_dir_path, args.out_path, 'run', args.model, args.dataset, start_time_date,
                              start_time_time)
    else:
        PREFIX = args.prefix
    model_save_dir, train_writer, eval_writer = init_method(PREFIX, args)
    data_path = os.path.join(file_dir_path, args.data_path)  # home/data


    if args.gpu != -1:
        gpus_per_node = torch.cuda.device_count()
    else:
        gpus_per_node = 1
    batch = args.batch_size * gpus_per_node
    lr = args.lr * gpus_per_node
    if args.dataset == "taobao":
        hist_itemfeature_dict_embedsize = {'hist_item_id': args.embed_size, 'hist_cate_id': args.embed_size}
        sparse_len = 3
        dense_len = 0
    elif args.dataset == "tmall":
        if args.only_idcat:
            hist_itemfeature_dict_embedsize = {'hist_item_id': args.embed_size, 'hist_cate_id': args.embed_size}
        else:
            hist_itemfeature_dict_embedsize = {'hist_item_id': args.embed_size, 'hist_cate_id': args.embed_size,
                                               'hist_merchant_id': args.embed_size,
                                               'hist_brand_id': args.embed_size}
        sparse_len = 7
        dense_len = 0
    elif args.dataset == "alipay":
        hist_itemfeature_dict_embedsize = {'hist_item_id': args.embed_size, 'hist_cate_id': args.embed_size,
                                           'hist_seller_id': args.embed_size}

        sparse_len = 4
        dense_len = 0
    else:
        raise NotImplementedError
    model = DIN(dataset_summary_file=os.path.join(data_path, args.dataset, 'feateng_data',
                                                  'summary_dict.pkl'), embed_size=args.embed_size,
                dnn_hidden_units=(200, 80), att_hidden_size=(64, 16),
                att_activation='Dice', att_weight_normalization=True,
                dnn_dropout=args.dnn_dropout,
                hist_itemfeature_dict_embedsize=hist_itemfeature_dict_embedsize, sparse_len=sparse_len,
                dense_len=dense_len)
    model = model.cuda() if args.gpu != -1 else model
    if args.use_ddp:
        # wrap the model
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    if args.use_mix_precision:
        scaler = GradScaler(enabled=args.use_mix_precision)
    else:
        scaler = None

    criterion = nn.BCEWithLogitsLoss()  #
    if args.optim == 'Adam':

        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args.wd)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=args.wd)
    else:
        raise NotImplementedError
    if args.decay_sched == 'Exponential':
        decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    elif args.decay_sched == 'CosineAnnealingWarmRestarts':
        decay = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    else:
        raise NotImplementedError

    if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
        logging.info("Model created.")

    # step 3: Load data
    train_loader, test_loader = DIEN_SequentialDataloader.load_data(
        dataset=args.dataset, batch_size=batch, ret_size=args.K, num_workers=args.num_workers, path=data_path,
        use_ddp=args.use_ddp,
        args=args
    )
    if (args.use_ddp is True and args.rank == 0) or args.use_ddp is False:
        logging.info("Data loaded.")

    # step 4: Training

    train_validate_din(args=args, model_save_dir=model_save_dir,
                       model=model, criterion=criterion,
                       optimizer=optimizer, decay=decay, train_loader=train_loader,
                       train_writer=train_writer,
                       test_loader=test_loader,
                       eval_writer=eval_writer, scaler=scaler)

    # test use the best model
    test_din(model=model, model_save_dir=model_save_dir, test_loader=test_loader, criterion=criterion,
             args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument("--dataset", type=str, help="Dataset to use, default: tmall")
    parser.add_argument("--data_path", type=str, help="Path to save the data")
    parser.add_argument("--out_path", type=str, help="Path to save the output")
    parser.add_argument("--seed", type=int, help="Random Seed")

    parser.add_argument("--model", type=str, help="Model Name")

    parser.add_argument("--K", type=int, help="Retrieval size.")

    parser.add_argument("--gpu", type=int, help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--use_ddp", action='store_true')
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")

    parser.add_argument("--batch_size", type=int, help="Batch size. Per Node")
    parser.add_argument("--wd", type=float, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, help="Learning Rate")
    parser.add_argument("--lr_decay", type=float, help="Exponential decay of learning rate")
    parser.add_argument('--decay_cycle', type=str
                        )
    parser.add_argument('--decay_sched', type=str)
    parser.add_argument("--embed_size", type=int)
    parser.add_argument("--num_workers", type=int, help="Number of processes to construct batches")
    parser.add_argument("--early_stop", type=int, help="Patience for early stop.")

    parser.add_argument("--dnn_dropout", type=float, help="Dropout.")
    parser.add_argument("--optim", type=str, help="Optimizer")
    parser.add_argument("--resume", action='store_true')

    parser.add_argument('--only_idcat', default=False,  #
                        action='store_true')  #
    parser.add_argument("--run_id", type=str, help="run_id")
    parser.add_argument("--baseline_auc", type=float, help="baseline_auc")
    # DDP参数·-----------------------------------------------------------------------------------------------------------
    parser.add_argument('--init_method',  #
                        help="init-method")  #
    parser.add_argument('-r', '--rank', type=int,  #
                        help='rank of current process')  #
    parser.add_argument('--world_size', type=int,  #
                        help="world size")  #
    parser.add_argument('--use_mix_precision', default=False,  #
                        action='store_true', help="whether to use mix precision")  #
    opts, _ = parser.parse_known_args()
    file_path = os.path.abspath(__file__)
    file_dir_path = os.path.dirname(file_path)
    if opts.config is not None:
        with open(os.path.join(file_dir_path, opts.config)) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open(os.path.join(file_dir_path,
                               'configs/hyper_sequential_din.json'), 'r',
                  encoding='UTF-8') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)
    torch.autograd.set_detect_anomaly(True)
    # np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    main(args)
