"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

H2GIGA_DIR='./../augmented_data/H2giga'

args = dict(

    cuda=False,

    save=True,
    save_dir='./exp',
    resume_path=None, 
    color_map={0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)},
    num_class = 5,
    train_dataset = {
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'train',
            'class_id':None,
            'size': None,
            'normalize':True,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'degrees': 90,
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 2,
        'workers': 1,
    }, 

    val_dataset = {
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 2,
        'workers': 1,
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [4,5]
        }
    }, 

    lr=5e-4,
    n_epochs=1,
    grid_size = 1024,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 2,
        'class_weight': [10, 10, 10, 10, 10],
        'num_class': 5
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 5,
    },
)


def get_args():
    return copy.deepcopy(args)
