#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:23:15 2022
# this is the file to load cifar 100 as id and corresponding oods and model from OpenOOD framework
@author: saiful
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
import pickle
import os
import sys
from pathlib import Path
from os import chdir
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print("current working directory @ easy_dev_load_data_n_model.py =>", os.getcwd())

old_path = Path.cwd()
print("old_path", old_path)
# change directory temporarily to OpenOOD
chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
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
config.num_workers = 8
config.save_output = False
config.parse_refs()

# %%
config
# change batch size from this directory /home/saiful/OpenOOD_framework/OpenOOD/configs/datasets/cifar10/cifar10.yml
# %%
# get dataloader
# returns dict_keys(['train', 'val', 'test'])
id_dataloader_from_openood_dict = get_dataloader(config)
print("id_dataloader_from_openood_dict", id_dataloader_from_openood_dict)

# init network
net = get_network(config.network).cuda()
# init ood evaluator
evaluator = get_evaluator(config)

# %%


def get_model_from_openood_for_imagenet():
    print("/OpenOOD/openood_id_ood_and_model_imagenet.py => get_model_from_openood_for_imagenet()")
    net = get_network(config.network).cuda()
    return net

# chdir(old_path)  # change again to the old directory

def id_dataloader_from_openood_repo_imagenet():
    print("/OpenOOD/openood_id_ood_and_model_imagenet.py => id_dataloader_from_openood_repo_imagenet()")
    train_loader=id_dataloader_from_openood_dict["train"]
    val_loader = id_dataloader_from_openood_dict["val"]
    test_loader = id_dataloader_from_openood_dict["test"]
    return train_loader, val_loader, test_loader  # ,

train_loader,val_loader,test_loader   = id_dataloader_from_openood_repo_imagenet()


#%%


def get_features_and_labels(dataloader, prev_value, curr_value):
    feature_list = []
    label_list = []
    image_dict = {}
    for i, image in enumerate(dataloader):
        print("i :",i)
        if i >= prev_value and i < curr_value:
            print(i)
            features = image['data']
            label = image['label']
            feature_list.append(features)
            label_list.append(label)
    image_dict['data'] = torch.cat(feature_list)
    image_dict['label'] = torch.cat(label_list)
    return image_dict

# value_1500 = get_features_and_labels(train_loader, 0, 1500)
# value_3000 = get_features_and_labels(train_loader, 1500, 3000)
# value_4500 = get_features_and_labels(train_loader, 3000, 4500)
# value_remain = get_features_and_labels(train_loader, 4500, 5100)
value_test = get_features_and_labels(test_loader, 0, 1500)
final_dataloader=value_test
# final_dataloader = {}
# final_dataloader.update(value_1500)
# final_dataloader.update(value_3000)
# final_dataloader.update(value_4500)
# final_dataloader.update(value_remain)
# val_dict = get_features_and_labels(val_loader)
# test_dict = get_features_and_labels(test_loader)
# test_features, test_labels = get_features_and_labels(test_loader)
# print("val_dict.keys() :", val_dict.keys())
print("len(final_dataloader['data']):",len(final_dataloader['data']))  


# with open('/data/saiful/imagenet_datasets2/val_loader_for_imagenet.pickle', 'wb') as handle:
#     pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/data/saiful/imagenet_datasets2/test_loader_for_imagenet.pickle', 'wb') as handle:
    pickle.dump(final_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('/data/saiful/imagenet_datasets2/test_loader_for_imagenet.pickle', 'wb') as handle:
#     pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("execution complete..")
#%%
"""
# %% ID data stuff
train_features_dict  = next(iter(train_loader))
val_features_dict = next(iter(val_loader))
# test_features_dict  = next(iter(test_loader))

# #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
print("train_features_dict.keys()",train_features_dict.keys())
print("val_features_dict.keys()",val_features_dict.keys())
# print("test_features_dict.keys()",test_features_dict.keys())

train_features = train_features_dict["data"]  # torch.Size([128, 3, 32, 32])
train_labels = train_features_dict["label"]  # torch.Size([128])

val_features=val_features_dict["data"]
val_labels=val_features_dict["label"]

# test_features=test_features_dict["data"]
# test_labels=test_features_dict["label"]

print("len(train_features):",len(train_features))
print("len(train_labels):",len(train_labels))

print("len(val_features):",len(val_features))
print("len(val_labels):",len(val_labels))

with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/full_val_loader_for_imagenet.pickle', 'wb') as handle:
    pickle.dump(val_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/data/saiful/imagenet_datasets2/train_loader_for_imagenet_full_train_images.pickle', 'wb') as handle:
    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("execution complete..")
# print("len(test_features):",len(test_features))
# print("len(test_labels):",len(test_labels))

# %%
ood = ood_dataloader_from_openood_repo_imagenet()
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
