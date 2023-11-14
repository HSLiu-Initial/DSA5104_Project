import configparser
import logging
import os
import sys

import numpy as np
import pandas as pd

from utils.es_writer import ESWriter

logging.basicConfig(level=logging.INFO)

# import logging
# logging.basicConfig(level=logging.INFO)




def merge_pool_file(base_folder, format, update_pool=False):
    if update_pool or (not (base_folder / 'search_pool' / format).exists()):
        print('Generating search pool...')
        X_ret = np.load(base_folder / 'train_input_all.npy')
        y_ret = np.load(base_folder / 'train_output_all.npy').reshape(-1, 1)
        ret_pool = np.concatenate([X_ret, y_ret], axis=1)
        pd.DataFrame(ret_pool, index=None).to_csv(
            os.path.join(base_folder, 'search_pool' + format), sep=',', header=False, index=False)
    else:
        print('Skip pool generation...')


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

    # ESWriter
    eswriter = ESWriter(os.path.join(root_dir_path, cnf.get(dataset, 'search_pool_file')),
                        dataset,
                        cnf.getint(dataset, 'sync_c_pos'))
    eswriter.write()
