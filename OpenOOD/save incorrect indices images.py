#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:06:57 2023

@author: saiful
"""
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
loaded_array = np.loadtxt('/home/saiful/confidence-magesh_MR/confidence-magesh/missing_indices.txt')

# Convert the NumPy array to a Python list
missing_indices = loaded_array.tolist()
missing_indices = [int(x) for x in missing_indices]
missing_indices=missing_indices[:20]
print(missing_indices)
# Create the directory for saving missing images if it doesn't exist

if not os.path.exists('/home/saiful/confidence-magesh_MR/confidence-magesh/missing_images'):
    os.makedirs('/home/saiful/confidence-magesh_MR/confidence-magesh/missing_images')

# Convert missing_indices to a set for faster lookup
missing_indices_set = set(missing_indices)

# Loop over the test dataset and save the images with indices in missing_indices
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

def ood_dataloader_from_openood_repo_mnist():
    print("/OpenOOD/openood_id_ood_and_model_mnist.py => ood_dataloader_from_openood_repo_mnist()")
    return get_ood_dataloader(config)


def save_images_in_folder(missing_indices,testloader):
    # Convert the NumPy array to a Python list
    missing_indices = loaded_array.tolist()
    missing_indices = [int(x) for x in missing_indices]
    missing_indices=missing_indices[:20]
    print(missing_indices)
    # Create the directory for saving missing images if it doesn't exist
    
    if not os.path.exists('/home/saiful/confidence-magesh_MR/confidence-magesh/missing_images'):
        os.makedirs('/home/saiful/confidence-magesh_MR/confidence-magesh/missing_images')
    
    # Convert missing_indices to a set for faster lookup
    missing_indices_set = set(missing_indices)


    print("save_images_in_folder()")
    missing_images = []
    for i, data in enumerate(testloader):
        image_tensor = data["data"]
        print("image_tensor.shape :", image_tensor.shape)
        for index in missing_indices_set:
            image = image_tensor[index]
            image = image.numpy()
            # Convert to numpy array and normalize pixel values to [0,1]
            image_array = (image.transpose(1, 2, 0) - image.min()) / (image.max() - image.min())
            plt.imsave(f'/home/saiful/confidence-magesh_MR/confidence-magesh/missing_images/missing_image_{index}.png', image_array)
            
    
chdir(old_path) # change again to the old directory

ood_dict_for_mnist = ood_dataloader_from_openood_repo_mnist()

print("ood_dict_for_mnist.keys():", ood_dict_for_mnist.keys()) #dict_keys(['val', 'nearood', 'farood'])
print("ood_dict_for_mnist[nearood].keys():",ood_dict_for_mnist["nearood"].keys()) # dict_keys(['cifar100', 'tin'])
print("ood_dict_for_mnist[farood].keys():",ood_dict_for_mnist["farood"].keys()) # dict_keys(['mnist', 'svhn', 'texture', 'place365'])


# access each dataloader
# fashionmnist_loader = ood_dict_for_mnist['nearood']['fashionmnist']
# notmnist_loader = ood_dict_for_mnist['nearood']['notmnist']

cifar10_testloader = ood_dict_for_mnist['farood']['cifar10']
# tin_loader = ood_dict_for_mnist['farood']['tin']
# places_loader = ood_dict_for_mnist['farood']['places365']
# texture_loader = ood_dict_for_mnist['farood']['texture']

save_images_in_folder(cifar10_testloader)

print("## execution complete")
