"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

H2GIGA_DIR='../Data/augmented/H2giga'

args = dict(

    cuda=True,

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
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'hs','instance', 'label'),
                        'type': (torch.FloatTensor,torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                            }
                },
                ]),
                },
            
            'batch_size': 15,
            'workers': 5,
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
                        'keys': ('image','hs','instance', 'label'),
                        'type': (torch.FloatTensor,torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                            }
                },
                ]),
                },
        'batch_size': 15,
        'workers': 5,
    }, 

    model = {
        'name': 'branced_hypernet', 
        'kwargs': {
            'in_channel': 164,
            'num_classes': [4,5]
        }
    }, 

    lr=5e-4,
    n_epochs=50,
    grid_size=1024,

    # loss options
    loss_opts={
        'class_weight': [10, 10, 10, 10, 10],
        'num_class': 5,
        'n_sigma': 2
    },
    
)


def get_args():
    return copy.deepcopy(args)
 