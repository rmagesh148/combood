#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:23:15 2022
# this is the file to load cifar 100 as id and corresponding oods and model from OpenOOD framework
@author: saiful
"""
import os
from pathlib import Path
from os import chdir

old_path = Path.cwd()
print("old_path", old_path)
# change directory temporarily to OpenOOD
chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
import pickle
import sys
import numpy as np
print("current working directory @ easy_dev_load_data_n_model.py =>", os.getcwd())
sys.path.insert(0, '..')  # Enable import from parent folder.

temp_path = Path.cwd()
#%%

# from openood.utils import config
# from openood.datasets import get_dataloader, get_ood_dataloader
# from openood.evaluators import get_evaluator
# from openood.networks import get_network
# import pickle
# import os, sys
# from pathlib import Path
# from os import chdir
# import numpy as np

# old_path=Path.cwd()
# print("old_path", old_path)
# # change directory temporarily to OpenOOD
# chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
# temp_path=Path.cwd()


#%%
# load config files for cifar10 baseline
config_files = [
    './configs/datasets/mnist/mnist.yml',
    './configs/datasets/mnist/mnist_ood.yml',
    './configs/networks/lenet.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)
# modify config 
# config.network.checkpoint = './results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
# config.network.checkpoint = './results/checkpoints/cifar10_res18_acc94.30.ckpt '
config.network.checkpoint = './results/checkpoints/mnist_lenet_acc98.50.ckpt' 
config.network.pretrained = True
config.num_workers = 8
config.save_output = False
config.parse_refs()

#%%
config
# change batch size from this directory /home/saiful/OpenOOD_framework/OpenOOD/configs/datasets/cifar10/cifar10.yml
#%%

# init network
net = get_network(config.network).cuda()
# init ood evaluator
evaluator = get_evaluator(config)

#%%
# get dataloader
id_dataloader_from_openood_dict = get_dataloader(config)    # returns dict_keys(['train', 'val', 'test'])
print("id_dataloader_from_openood_dict",id_dataloader_from_openood_dict)
ood_dataloader_from_openood_dict = get_ood_dataloader(config)
print("ood_dataloader_from_openood_dict",ood_dataloader_from_openood_dict)

def get_model_from_openood_for_mnist():
    print("/OpenOOD/openood_id_ood_and_model_mnist.py => get_model_from_openood_for_mnist()")
    net =get_network(config.network).cuda()
    return net

#%%
# id_dataloader_from_openood_dict 


def id_dataloader_from_openood_repo_mnist():
    print("/OpenOOD/openood_id_ood_and_model_mnist.py => id_dataloader_from_openood_repo_mnist()")
    train_loader=id_dataloader_from_openood_dict["train"]
    val_loader = id_dataloader_from_openood_dict["val"]
    test_loader=id_dataloader_from_openood_dict["test"]   
    return train_loader,val_loader, test_loader

# r = id_dataloader_from_openood_repo_mnist() 

def ood_dataloader_from_openood_repo_mnist():
    print("/OpenOOD/openood_id_ood_and_model_mnist.py => ood_dataloader_from_openood_repo_mnist()")
    return get_ood_dataloader(config)

chdir(old_path) # change again to the old directory
#%% saving mnist dataloader as pickle

# with open('./dataloader pickle/mnist28 pickle loaders/id_train_mnist28_dataloader.pickle', 'wb') as handle:
#     pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./dataloader pickle/mnist28 pickle loaders/id_val_mnist28_dataloader.pickle', 'wb') as handle:
#     pickle.dump(val_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./dataloader pickle/mnist28 pickle loaders/id_test_mnist28_dataloader.pickle', 'wb') as handle:
#     pickle.dump(test_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% saving ood as pickle file
# ood_dict_for_mnist=ood_dataloader_from_openood_repo_mnist()