import math
import cv2
import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np


def psnr(img1_path,img2_path):
    original = cv2.imread(img1_path)
    contrast = cv2.imread(img2_path,1)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1_path,img2_path):
    npImg1 = cv2.imread(img1_path)
    npImg2 = cv2.imread(img2_path)

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()


    img1 = Variable( img1,  requires_grad=False)
    img2 = Variable( img2, requires_grad = True)

    print(img1.shape)
    print(img2.shape)
    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = 1-pytorch_ssim.ssim(img1, img2).item()
    print("Initial ssim:", ssim_value)
    return ssim_value