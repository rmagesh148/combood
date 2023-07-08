#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:41:19 2023

@author: saiful
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
sys.path.insert(0, '..')  # Enable import from parent folder.
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.parallel
import torch.utils.data
from models.resnet_react import resnet50
# from models.document import resnet50, resnet50_from_torch_hub
# from torchvision import models
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from torch.utils.data import RandomSampler, DataLoader, Subset,SubsetRandomSampler#,RandomSubsetSampler
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
batchsize=32
num_train_samples= 319837  #319837
num_val_samples= 39995 #39995
num_test_samples= 39996   #39996


print(f"flag 1.65 The document dataset is now running with \n num_train_samples:{num_train_samples},\n num_val_samples:{num_val_samples},\n num_test_samples:{num_test_samples}"  )
def load_document_id_data():
    print("document_id_ood_n_model_loader.py =>  load_document_id_data()")


    # check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    transform=transforms.Compose([
                                   transforms.ToTensor(), 
                                   transforms.Resize((224,224)),
                                    transforms.Normalize(0.9199, 0.1853),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                   # transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                  ])

    # defining dataset path
    path_trainimages=r"/data/saiful/rvl-cdip old/train/"
    path_testimages=r"/data/saiful/rvl-cdip old/test/"
    path_validationimages=r"/data/saiful/rvl-cdip old/validation"


    train_dataset_docu = torchvision.datasets.ImageFolder(root=path_trainimages,                                           
                                                     transform=transform)
    test_dataset_docu = torchvision.datasets.ImageFolder(root=path_testimages,                                           
                                                     transform=transform)
    validation_dataset_docu = torchvision.datasets.ImageFolder(root=path_validationimages,                                           
                                                     transform=transform)


    print(train_dataset_docu.class_to_idx)

    print("\ntrain_dataset_docu length :", len(train_dataset_docu))
    print("validation_dataset_docu length:",len(validation_dataset_docu))
    print("test_dataset_docu length :", len(test_dataset_docu))


    #Random Sampling data
    # torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)


    #train
    random_sampled_train_set_docu = torch.utils.data.RandomSampler(train_dataset_docu, 
                                                                    replacement=False, 
                                                                    num_samples=num_train_samples, 
                                                                    generator=None)

    random_sampled_train_set_docu_dataloader  = torch.utils.data.DataLoader(
                                                dataset=train_dataset_docu, 
                                                 sampler=random_sampled_train_set_docu,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=0)

    # val
    random_sampled_val_set_docu = torch.utils.data.RandomSampler(validation_dataset_docu, 
                                                                    replacement=False, 
                                                                    num_samples=num_val_samples, 
                                                                    generator=None)

    random_sampled_val_set_docu_dataloader  = torch.utils.data.DataLoader(
                                                dataset=validation_dataset_docu, 
                                                 sampler=random_sampled_val_set_docu,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=0)
    # test
    random_sampled_test_set_docu = torch.utils.data.RandomSampler(test_dataset_docu, 
                                                                    replacement=False, 
                                                                    num_samples=num_test_samples, 
                                                                    generator=None)


    random_sampled_test_set_docu_dataloader  = torch.utils.data.DataLoader(
                                                dataset=test_dataset_docu, 
                                                 sampler=random_sampled_test_set_docu,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=0)

        
    print("working on len(random_sampled_train_set_docu):",len(random_sampled_train_set_docu))
    print("working on len(random_sampled_val_set_docu):",len(random_sampled_val_set_docu))
    print("working on len(random_sampled_test_set_docu):",len(random_sampled_test_set_docu))
    train_loader = random_sampled_train_set_docu_dataloader
    # print("train_loader.shape:",train_loader.shape)
    # print("train_loader[0].shape:",train_loader[0][0].shape)
    # for idx, (images, labels) in enumerate(train_loader):
    #     print("images.shape:", images.shape)
    #     print("labels.shape:", labels.shape)
    #     print("idx:", idx)

    val_loader = random_sampled_val_set_docu_dataloader
    test_loader = random_sampled_test_set_docu_dataloader
    print("returning Train, Val and Test Set for document dataset \n")

    # return {"Train": train_loader, "Val": val_loader, "Test": test_loader}
    return {"Train": train_loader,
            "Val": val_loader,
            "Test": test_loader}




# =============================================================================
# document ood 
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_list = os.listdir(data_dir)
        print()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
    
def load_document_ood_data():
    print("document_id_ood_n_model_loader.py =>  load_document_ood_data():")
    # Define the path to the folder containing the images
    data_path_rvl_cdip_n = "/data/saiful/rvl-cdip-ood/rvl-cdip-ood/RVL-CDIP-N/"
    data_path_rvl_cdip_o = "/data/saiful/rvl-cdip-ood/rvl-cdip-ood/RVL-CDIP-O/"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.9199, 0.1853),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    dataset_rvl_cdip_n = CustomDataset(data_dir=data_path_rvl_cdip_n, transform=transform)
    rvl_cdip_n_loader = DataLoader(dataset_rvl_cdip_n, batch_size=32, shuffle=False)

    dataset_rvl_cdip_o = CustomDataset(data_dir=data_path_rvl_cdip_o, transform=transform)
    rvl_cdip_o_loader = DataLoader(dataset_rvl_cdip_o, batch_size=32, shuffle=False)
    
    print('len(rvl_cdip_n_loader.dataset):',len(rvl_cdip_n_loader.dataset))
    print('len(rvl_cdip_o_loader.dataset):',len(rvl_cdip_o_loader.dataset))
    

    ood_datasets = {
        "rvl_cdip_n" : rvl_cdip_n_loader,
        "rvl_cdip_o" : rvl_cdip_o_loader,
    }
    return ood_datasets

def load_document_rvl_cdip_o_CustomDataset():
    print("document_id_ood_n_model_loader.py =>  load_document_ood_data():")
    # Define the path to the folder containing the images
    data_path_rvl_cdip_n = "/data/saiful/rvl-cdip-ood/rvl-cdip-ood/RVL-CDIP-N/"
    data_path_rvl_cdip_o = "/data/saiful/rvl-cdip-ood/rvl-cdip-ood/RVL-CDIP-O/"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.9199, 0.1853),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    # dataset_rvl_cdip_n = CustomDataset(data_dir=data_path_rvl_cdip_n, transform=transform)
    # rvl_cdip_n_loader = DataLoader(dataset_rvl_cdip_n, batch_size=32, shuffle=False)

    dataset_rvl_cdip_o = CustomDataset(data_dir=data_path_rvl_cdip_o, transform=transform)
    # rvl_cdip_o_loader = DataLoader(dataset_rvl_cdip_o, batch_size=32, shuffle=False)
    
    # print('len(rvl_cdip_n_loader.dataset):',len(rvl_cdip_n_loader.dataset))
    # print('len(rvl_cdip_o_loader.dataset):',len(rvl_cdip_o_loader.dataset))
    

    # ood_datasets = {
    #     "rvl_cdip_n" : rvl_cdip_n_loader,
    #     "rvl_cdip_o" : rvl_cdip_o_loader,
    # }
    return dataset_rvl_cdip_o
# =============================================================================
# load pretrained model
# =============================================================================
def load_resnet50_model_for_document_dataset():
    print("document_id_ood_n_model_loader.py =>  load_resnet50_model_for_document_dataset()")
    # model_ft = resnet50.ResNet50(pretrained=False)
    # model_ft = resnet50.ResNet50()

    # model_ft.fc = nn.Sequential(
    #                 nn.Linear(2048, 128),
    #                 nn.ReLU(inplace=True),
    #                 nn.Linear(128, 16)).to(device)
    
    # print("model_ft:", model_ft)
    
    ##
    # model = resnet50_from_torch_hub.ResNet50(pretrained=False)
    model = resnet50(pretrained=False, num_classes=2048)  # interms of resnet_react.py

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
    model.fc = torch.nn.Linear(in_features=2048, out_features=16, bias=True)
    print("model resnet50 from hub :", model)
    print("model:", model)
    
    model_ft = model
    model_ft = model_ft.to(device)  


    # checkpoint = "/home/saiful/confidence-magesh_MR/confidence-magesh/document classification/saved trained models/resnet50/resnet50_model_epoch_20_on_319837_trainimages.ckpt"
    # checkpoint = "/data/saiful/document classification/saved trained models/resnet50_checkpoints/resnet50_acc0.9_epoch40_on_319837_trainimages_load.ckpt"
    checkpoint = "/home/saiful/confidence_icdb/confidence-magesh/document classification/saved trained models/resnet50_checkpoints/resnet50_acc0.9_epoch40_on_319837_trainimages_load.ckpt"
    
    state_dict = torch.load(checkpoint, map_location=device)     
    model_ft.load_state_dict(state_dict, strict=True)      
    m= model_ft       
    transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    return m.eval() , transform


# calling id dataloader
# x,y,z = load_document_id_data()
# calling ood dataloader
# y = load_document_ood_data()
# # calling trained resnet50 model 

# m , t=load_resnet50_model_for_document_dataset()


# a=load_document_rvl_cdip_o_CustomDataset()
