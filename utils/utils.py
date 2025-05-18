import torch

import torch.backends.cudnn as cudnn
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def save_imgs( msk_pred, img_name, save_path, datasets, threshold=0.5):
    if os.path.exists(save_path + str(img_name) +'.png'):
        return
    if datasets == 'retinal':
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    if not os.path.exists(''):
        os.makedirs('')
    plt.imsave('' + str(img_name) + '.png', msk_pred , cmap='gray')
