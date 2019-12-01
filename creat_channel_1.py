import torch
import torchvision
import numpy as np
from PIL import Image
import os 

# https://github.com/wuhuikai/DeepGuidedFilter/tree/master/GuidedFilteringLayer/GuidedFilter_PyTorch
from guided_filter_pytorch.guided_filter import GuidedFilter as SoftMatting

#Display = lambda img_tensor: display(torchvision.transforms.ToPILImage()(img_tensor).convert("RGB"))



#dark channel

def get_darkChannel(image, patch=15):
    intensity = -torch.nn.MaxPool2d(patch, padding=patch//2, stride=1)(-image)
    return torch.min(torch.min(intensity[:, 0], intensity[:, 1]), intensity[:, 2])


#load img data
path = 'ori'
files = os.listdir(path)
imageName = []
for file in files:
    if not os.path.isdir(file):
        imageName.append(path+"/"+file)


for i in range(len(imageName)):
    output = "darkimg/dark_"+imageName[i].split("/")[1]
    image_raw = Image.open(imageName[i]).convert('RGB')
    image_tensor = torchvision.transforms.ToTensor()(image_raw)
    image_batch = image_tensor.unsqueeze(0)
    darkChannel = get_darkChannel(image_batch)
    darkChannel_batch = darkChannel.unsqueeze(0)
    assert([i == j for i,j in zip(image_tensor.shape, darkChannel.shape)])
    image = torchvision.transforms.ToPILImage()(darkChannel)
    image.save(output)



# darkChannel = get_darkChannel(image_batch)
# darkChannel_batch = darkChannel.unsqueeze(0)

# assert([i == j for i,j in zip(image_tensor.shape, darkChannel.shape)])

# image = torchvision.transforms.ToPILImage()(darkChannel)
# image.save("w.jpg")

#Display(darkChannel)