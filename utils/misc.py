import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_npz(epoch, log_dict, results_folder, savename='train'):
    """
    From: https://github.com/c-rbp/pathfinder_experiments/blob/main/utils/misc_functions.py
    """
    with open(results_folder + savename + '.npz', 'wb') as f:
        np.savez(f, **log_dict)


def remove_module(model_path):
    state_dict = torch.load(model_path)['state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name != 'unit1.mult':
            new_state_dict[name] = v
    return new_state_dict


def save_checkpoint(state, key, results_folder):
    save_folder = results_folder + 'saved_models/'
    try:
        os.mkdir(save_folder)
    except:
        pass
    filename = save_folder + 'model_{0}_{1:04d}_epoch_{2:02d}_checkpoint.pth.tar'.format(key,
                                                                                         int(state[key] * 1e4),
                                                                                         state['epoch'])
    torch.save(state, filename)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    From: https://github.com/c-rbp/pathfinder_experiments/blob/main/utils/misc_functions.py

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n != 0:  # [LG] added this condition
            self.history.append(val)
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count