# -*- coding:utf-8 -*-

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use, default: tmall")
    opts, _ = parser.parse_known_args()
    os.system("python 1_util_feateng.py " + opts.dataset)
    os.system("python 2_util_hori_split.py " + opts.dataset)
    os.system("python 3_util_get_dataset_sum.py " + opts.dataset)
    # os.system("cd ..")  #
    os.system("python 4_rim_insert_es.py " + opts.dataset)
    os.system("python 4_dien.py " + opts.dataset)
    os.system("python 5_rim_pre_search_new.py " + opts.dataset + " 2000 10 RIM")

    os.system("python 5_dien.py " + opts.dataset + " 100")
