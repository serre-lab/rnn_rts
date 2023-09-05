import os

from .cocodots import CocoDots
from .mazes import Mazes
from .dataset_str_mapping import MAPPING


def setup_dataset(dataset_str, data_root, subset, shuffle, **kwargs):

    if dataset_str  in ['cocodots_train', 'cocodots_train_light', 'cocodots_val', 'cocodots_val_mini']:
        h = kwargs.get('base_size', 150)  # input image height
        w = kwargs.get('base_size', 150)  # input image width
        dataset = CocoDots(MAPPING[dataset_str],
                           os.path.join(data_root,
                                        {'cocodots_train': 'train2017',
                                         'cocodots_train_light': 'train2017',
                                         'cocodots_val': 'val2017',
                                         'cocodots_val_mini': 'val2017_mini'}[dataset_str]),
                           size=(h, w),
                           subset=subset,
                           shuffle=shuffle)

    elif dataset_str in ['mazes_train', 'mazes_val']:
        dataset = Mazes(MAPPING[dataset_str],
                        data_root,
                        subset=subset,
                        shuffle=shuffle)

    else:
        raise NotImplementedError("This dataset_str is not implemented.")

    return dataset
