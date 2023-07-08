#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:23:15 2022
# this is the file to load cifar 100 as id and corresponding oods and model from OpenOOD framework
@author: saiful
"""
# from openood.utils import config
# from openood.datasets import get_dataloader, get_ood_dataloader
# from openood.evaluators import get_evaluator
# from openood.networks import get_network
# import pickle
# import os
# import sys
# from pathlib import Path
# from os import chdir
# import numpy as np
# print("current working directory @ easy_dev_load_data_n_model.py =>", os.getcwd())

# old_path = Path.cwd()
# print("old_path", old_path)
# # change directory temporarily to OpenOOD
# chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
# temp_path = Path.cwd()
#%%
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
# %%
# load config files for cifar10 baseline
config_files = [
    './configs/datasets/imagenet/imagenet.yml',
    './configs/datasets/imagenet/imagenet_ood.yml',
    './configs/networks/resnet50.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)
# modify config
# config.network.checkpoint = './results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
# config.network.checkpoint = './results/checkpoints/cifar10_res18_acc94.30.ckpt '
config.network.checkpoint = './results/checkpoints/imagenet_res50_acc76.10.pth'
config.network.pretrained = True
config.num_workers = 32
config.save_output = False
config.parse_refs()

# %%
config
# change batch size from this directory /home/saiful/OpenOOD_framework/OpenOOD/configs/datasets/cifar10/cifar10.yml
# %%
# get dataloader
# returns dict_keys(['train', 'val', 'test'])
id_dataloader_from_openood_dict = get_dataloader(config)
# print("id_dataloader_from_openood_dict", id_dataloader_from_openood_dict)
ood_dataloader_from_openood_dict = get_ood_dataloader(config)
# print("ood_dataloader_from_openood_dict", ood_dataloader_from_openood_dict)
# init network
net = get_network(config.network).cuda()
# init ood evaluator
evaluator = get_evaluator(config)

# %%
def get_model_from_openood_for_imagenet():
    print("/OpenOOD/openood_id_ood_and_model_imagenet.py => get_model_from_openood_for_imagenet()")
    net = get_network(config.network).cuda()
    return net
#%%
def id_dataloader_from_openood_repo_imagenet():
    print("/OpenOOD/openood_id_ood_and_model_imagenet.py => id_dataloader_from_openood_repo_imagenet()")
    train_loader=id_dataloader_from_openood_dict["train"]
    val_loader = id_dataloader_from_openood_dict["val"]
    test_loader = id_dataloader_from_openood_dict["test"]
    return train_loader, val_loader, test_loader  # ,

train_loader,val_loader,test_loader   = id_dataloader_from_openood_repo_imagenet()

def ood_dataloader_from_openood_repo_imagenet():
    print("/OpenOOD/openood_id_ood_and_model_imagenet.py => ood_dataloader_from_openood_repo_imagenet()")
    return get_ood_dataloader(config)

chdir(old_path)


# chdir(old_path)  # change again to the old directory


# def id_dataloader_from_openood_repo_imagenet():
#     print("/OpenOOD/openood_id_ood_and_model_imagenet.py => id_dataloader_from_openood_repo_imagenet()")
#     # train_loader=id_dataloader_from_openood_dict["train"]
#     val_loader = id_dataloader_from_openood_dict["val"]
#     test_loader = id_dataloader_from_openood_dict["test"]
#     train_loader = val_loader
#     print("## ** using val_loader as train_loader")
#     return train_loader, val_loader, test_loader  # ,

r = id_dataloader_from_openood_repo_imagenet()


# %% ID data stuff
# train_features_dict  = next(iter(train_loader))
# val_features_dict = next(iter(val_loader))
# test_features_dict  = next(iter(test_loader))

# #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
# print("train_features_dict.keys()",train_features_dict.keys())
# print("val_features_dict.keys()",val_features_dict.keys())
# print("test_features_dict.keys()",test_features_dict.keys())

# train_features = train_features_dict["data"]  # torch.Size([128, 3, 32, 32])
# train_labels = train_features_dict["label"]  # torch.Size([128])

# val_features=val_features_dict["data"]
# val_labels=val_features_dict["label"]

# test_features=test_features_dict["data"]
# test_labels=test_features_dict["label"]

# print("len(train_features):",len(train_features))
# print("len(train_labels):",len(train_labels))

# print("len(val_features):",len(val_features))
# print("len(val_labels):",len(val_labels))


# print("len(test_features):",len(test_features))
# print("len(test_labels):",len(test_labels))
# %%
"""ood = ood_dataloader_from_openood_repo_imagenet()
ood.keys()
ood['val']    
ood['nearood']    
ood['farood']    
cifar100_loader = ood['nearood']['cifar100']
cifar100_features_dict  = next(iter(cifar100_loader))
p =cifar100_features_dict

print("cifar100_features_dict.keys():",cifar100_features_dict.keys())
# >>> cifar100_features_dict.keys(): dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
cifar100_imgtensr = cifar100_features_dict["data"]  # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
cifar100_label = cifar100_features_dict["label"]  # cifar100_label.shape = torch.Size([128])
cifar100_imgnp= cifar100_imgtensr.numpy()  # shape (128, 3, 32, 32)
cifar100_imgnp = np.moveaxis(cifar100_imgnp, 1, 3)  # converting to (128, 32, 32, 3) 
# df_cifar100 = scale_and_save_in_df(cifar100_imgnp, np.nan, scale=True)


"""
