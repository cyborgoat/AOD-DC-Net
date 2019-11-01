import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
class MyDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
        
    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = torchvision.transforms.ToTensor()(image)
        return image

def parse_data(datadir):
    img_list = []
    features = []

    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                features.append(os.path.splitext(filename)[0])
    img_list.sort()
    features.sort()
    ratio = 0.7
    num_sample_train = int(len(img_list)*ratio)
    num_sample_val = int(len(img_list)-num_sample_train)
    
    train_inputs = []
    val_inputs = []
    features_train = []
    features_val =[]
    
    for i in img_list[:num_sample_train]:
        train_inputs.append(i)
    for i in img_list[num_sample_train:]:
        val_inputs.append(str(i))
    for i in features[:num_sample_train]:
        features_train.append(str(i))
    for i in features[num_sample_train:]:
        features_val.append(str(i))
    
    
    print('{}\t\t{}\n'.format('#Images', len(img_list)))
    print('{}\t\t{}\n'.format('#Images train', len(train_inputs)))
    print('{}\t\t{}\n'.format('#Images val', len(val_inputs)))
    return train_inputs, val_inputs, features_train, features_val

train_list,val_list,train_features,val_features = parse_data('/Users/chen/Desktop/part1')
#clear_train,clear_val,cleartrain_features,clearval_features = parse_data(‘’)

trainset= MyDataset(train_list)
valset = MyDataset(val_list)
#cleartrain = MyDataset(clear_train)
#clearval = MyDataset(clear_val)

trainloader = DataLoader(trainset, batch_size=64, shuffle=False, drop_last=False)
valloader = DataLoader(valset, batch_size=64, shuffle=False, drop_last=False)
#cleartrainloader = DataLoader(cleartrain, batch_size=64, shuffle=False, drop_last=False)
#clearvalloader = DataLoader(clearval, batch_size=64, shuffle=False, drop_last=False)


