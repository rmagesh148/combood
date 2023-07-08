#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:49:41 2023

@author: saiful
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:55:27 2022

@author: msajol1
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#%% import module
import os
import torch
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.parallel
import torch.utils.data
from torchvision import models
import torch.utils.data as data
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
import torch.nn as nn
import tqdm
import numpy as np
import pandas as pd
import glob
import math
import random
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import RandomSampler, DataLoader, Subset,SubsetRandomSampler#,RandomSubsetSampler
import time

model_name = 'resnet18'
path = Path(f"./document classification/saved trained models/{model_name}")
path.mkdir(exist_ok=True, parents=True)
#%% variables and transform
batchsize=100
epochs = 100 # Number of epochs
num_train_samples= 5000  #319837
num_val_samples= 1000  #39995
num_test_samples= 1000   #39996

# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
transform =transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                       std=[0.229, 0.224, 0.225]),
       # transforms.Normalize((0, 0, 0), (1, 1, 1)),
])

#%%  defining dataset path
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

#%% Random Sampling data
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

# Validation
train_targets = []
for _, target in random_sampled_train_set_docu_dataloader:
    train_targets.append(target)
train_targets = torch.cat(train_targets)

print("\n Class labels and their corresponding counts:")
print(train_targets.unique(return_counts=True))
print("len(random_sampled_train_set_docu):",len(random_sampled_train_set_docu))
print("len(random_sampled_val_set_docu):",len(random_sampled_val_set_docu))
print("len(random_sampled_test_set_docu):",len(random_sampled_test_set_docu))

#%%
def show_transformed_images(dataset):
    loader= torch.utils.data.DataLoader(dataset=dataset, batch_size=40,  shuffle=True)
    batch=next(iter(loader))
    images, label = batch 
    
    grid=torchvision.utils.make_grid(images)
    plt.figure()
    plt.imshow(np.transpose(grid, (1,2,0)))
    print("\nlabels of {} {} " .format(dataset,label))
    print("label of the images {}".format(label))
    
# show_transformed_images(train_set_cifar)                                        
# show_transformed_images(subset_train_docu_by_class)                                         


#%% model 
# model_ft = models.resnet50(pretrained=True)  #models.densenet161()
# model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# calling mobilenet
# model_ft = models.mobilenet_v2(pretrained=True)
# model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=16)

# calling resnet50
model_ft = models.resnet50(pretrained=True)
model_ft.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 16)).to(device)

#
# model = ImageClassifier(backbone="mobilenet_v2", num_classes= 16)

# load the model onto the device(GPU)
model_ft = model_ft.to(device)

# Freeze model weights
for param in model_ft.parameters():
    param.requires_grad = True   # False-> feature extracting training only last layers ,  requires_grad=False is "untrainable" or "frozen" in place.
    # True -> training from scratch or finetuning.
# Print a summary using torchinfo (uncomment for actual output)

#%% criterion and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer  = optim.Adam(model_ft.classifier.parameters(),lr = 0.001)
optimizer  = optim.Adam(model_ft.parameters(),lr = 0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.35)


# =============================================================================
# training loop
# =============================================================================

history=[]
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []
epoch_list=[]
print("Training started..")
for epoch in range(epochs):
    
    epoch_start = time.time()
    
    # print("Epoch: {}/{}".format(epoch+1, epochs))
    # Set to training mode
    model_ft.train()
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

    # training on trainloader
    for i, (inputs, labels) in enumerate(random_sampled_train_set_docu_dataloader):
        print('i:',i)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Clean existing gradients
        optimizer.zero_grad()
        # Forward pass - compute outputs on input data using the model
        outputs = model_ft(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        
        #
        # loss.requires_grad = True
        #
        
        # Backpropagate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
        # print(" Training Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
        
        # print("train predictions:",predictions)
        # print("train labels:", labels)


    # Validation - No gradient tracking needed
    with torch.no_grad():
        # Set to evaluation mode
        model_ft.eval()
        # Validation loop
        for j, (inputs, labels) in enumerate(random_sampled_val_set_docu_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass - compute outputs on input data using the model
            outputs = model_ft(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)
            # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
            
    # Find average training loss and training accuracy per epoch
    avg_train_loss = train_loss/len(random_sampled_train_set_docu) 
    avg_train_acc = train_acc/float(len(random_sampled_train_set_docu))
    
    # Find average validation loss and validation accuracy per epoch
    avg_valid_loss = valid_loss/len(random_sampled_val_set_docu) 
    avg_valid_acc = valid_acc/float(len(random_sampled_val_set_docu))
    
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    valid_loss_list.append(avg_valid_loss)
    valid_acc_list.append(avg_valid_acc)
    epoch_list.append(epoch+1)
    # history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    # print("\n ##  Training and validation loss and  accuracy per epoch")
    print("\nEpoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    # Save if the model has best accuracy till now
    # torch.save(model_ft.state_dict(), "/home/saiful/confidence-magesh_MR/confidence-magesh/document classification/saved trained models/mobilenet/" 'mobilenet'+'_model_'+'epoch'+str(epoch)+'.pt')
    torch.save(model_ft.state_dict(), path/ f"{model_name}_model_epoch_{epoch}.pt")
    epoch+=1
    if epoch== epochs-1:
        print("flag1")
        print("Last epoch : ", epoch)
        break
print("Training Finished")

#%%
print("len(train_acc_list) ",len(train_acc_list))
print("len(valid_acc_list) ",len(valid_acc_list))

# plotting  accuracy
plt.figure(figsize=(16, 9))
plt.plot(epoch_list, train_acc_list)
plt.plot( epoch_list,valid_acc_list)
plt.legend([ "train_acc",'valid_acc' ])
plt.xlabel('Epoch Number')
plt.ylabel('accuracy')

plt.savefig("document_dataset "+'_accuracy_curve.png')
plt.show()

#%%
# plotting losses 
plt.figure(figsize=(16, 9))
plt.plot( epoch_list, train_loss_list)
plt.plot( epoch_list, valid_loss_list)
plt.legend([ "train_loss",'valid_loss' ])
plt.xlabel('Epoch Number')
# plt.ylabel('Loss')

plt.savefig("document_dataset "+'_loss_curve.png')
plt.show()

#%%

