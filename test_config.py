"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy

import torch
from utils import transforms as my_transforms

H2GIGA_DIR='../augmented_data/H2giga'


args = dict(

    cuda=False,
    display=True,

    save=True,
    save_dir='./exp_20220715_(1)',
    checkpoint_path='./best_iou_model.pth',
    color_map={0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)},
    num_class = 5,
    dataset= { 
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'test_1',
            'class_id': None,            
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [4, 5],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
