import math
import cv2
import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
from skimage.measure import compare_ssim
import argparse
import imutils


def psnr(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path,1)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1_path,img2_path):
    # 2. Construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
    # ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
    # args = vars(ap.parse_args())

    # 3. Load the two input images
    imageA = cv2.imread(img1_path)
    imageB = cv2.imread(img2_path)

    # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 6. You can print only the score if you want
    print("SSIM: {}".format(score))

if __name__ == "__main__":
    psnr_result = psnr("imgs/foggy_demo1.jpg","imgs/foggy_demo1.jpg")
    print(psnr_result)
    ssim_result = ssim("imgs/foggy_demo1.jpg","imgs/foggy_demo1.jpg")
    print(ssim_result)