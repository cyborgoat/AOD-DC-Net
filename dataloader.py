#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:27:02 2019

@author: chen
"""

import os
import sys
import torch

import numpy as np
from PIL import Image
import glob
import random
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision

def split_validation(image,haze):
    train_list = []
    val_list = []
    
    hazeimage = glob.glob(haze+"*.jpg")
    
    tmp_dict = {}
    
    for i in hazeimage:
        
        i = i.split("/")[-1]
        key = i.split("_")[1] + "_" + image.split("_")[2] 
        
        if key in tmp_dict.keys():
            tmp_dict[key].append(i)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(i)
    
    train_key = []
    val_key = []
    
    length = len(tmp_dict.keys())
    ratio = 0.8
    for i  in range(length):
        if i<length*ratio:
            train_key.append(list(tmp_dict.keys())[i])
        else:
            val_key.append(list(tmp_dict.keys())[i])
    
    for key in list(tmp_dict.keys()):
        if key in train_key:
            for j in tmp_dict[key]:
                train_list.append([image + key, haze + j])
        
        else:
            for j in tmp_dict[key]:
                val_list.append([image + key, haze + j])
        
    random.shuffle(train_list)
    random.shuffle(val_list)
    
    return train_list, val_list

class DataLoader(Dataset):
    
    def __init__(self,image,haze,mode = "train"):
        self.train_inputs,self.val_inputs = split_validation(image,haze)
        
        if mode =="train":
            self.img_list = self.train_inputs
            print('{}\t\t{}\n'.format('#Images', len(self.train_inputs)))
        
        else:
            self.img_list = self.val_inputs
            print('{}\t\t{}\n'.format('#Images', len(self.val_inputs)))
        
    def __getitem__(self, idx):
        image,haze = self.img_list[idx]
        
        clear_image = Image.open(image).convert('RGB')
        blur_image = Image.open(haze).convert('RGB')
        
        clear_image = torchvision.transforms.ToTensor()(clear_image)
        blur_image = torchvision.transforms.ToTensor()(blur_image)
        
        return clear_image, blur_image
    
    def __len__(self):
        return len(self.img_list)

