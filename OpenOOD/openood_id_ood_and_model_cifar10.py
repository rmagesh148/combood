# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
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
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/resnet18_32x32.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)
# modify config
# config.network.checkpoint = './results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt'
# config.network.checkpoint = './results/checkpoints/cifar10_res18_acc94.30.ckpt '
config.network.checkpoint = '/home/saiful/confidence-magesh_MR/confidence-magesh/models/cifar10/resnet/cifar10_res18_acc94.30.ckpt'
config.network.pretrained = True
config.num_workers = 8  # default 8
config.save_output = False
config.parse_refs()

# %%
config
# change batch size from this directory /home/saiful/OpenOOD_framework/OpenOOD/configs/datasets/cifar10/cifar10.yml
# %%
# get dataloader
id_dataloader_from_openood_dict = get_dataloader(config)    # returns dict_keys(['train', 'val', 'test'])
print("id_dataloader_from_openood_dict",id_dataloader_from_openood_dict)
# ood_dataloader_from_openood_dict = get_ood_dataloader(config)
# print("ood_dataloader_from_openood_dict",ood_dataloader_from_openood_dict)
# init network

net = get_network(config.network).cuda()
# init ood evaluator
evaluator = get_evaluator(config)

# %%

def get_model_from_openood():
    get_network(config.network).cuda()
    return net

#%%
# id_dataloader_from_openood_dict 
train_loader=id_dataloader_from_openood_dict["train"]
val_loader = id_dataloader_from_openood_dict["val"]
test_loader=id_dataloader_from_openood_dict["test"]

def id_dataloader_from_openood_repo_cifar10():
    print("/OpenOOD/openood_id_ood_and_model.py => id_dataloader_from_openood_repo_cifar10()")
    train_loader=id_dataloader_from_openood_dict["train"]
    val_loader = id_dataloader_from_openood_dict["val"]
    test_loader=id_dataloader_from_openood_dict["test"]   
    return train_loader,val_loader, test_loader

# r = id_dataloader_from_openood_repo_cifar10() 

def ood_dataloader_from_openood_repo_cifar10():
    print("/OpenOOD/openood_id_ood_and_model.py => ood_dataloader_from_openood_repo_cifar10()")
    return get_ood_dataloader(config)

chdir(old_path)