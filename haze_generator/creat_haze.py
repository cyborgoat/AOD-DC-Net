import cv2
import numpy as np 
import random
import os

path = 'part10'
files = os.listdir(path)
imageName = []
for file in files:
    if not os.path.isdir(file):
        imageName.append(path+"/"+file)



for i in range(len(imageName)):
	#print(imageName[i])
	output = "output/150_"+imageName[i].split("/")[1]
	img = cv2.imread(imageName[i],1)
	#print(type(img))
	imgInfo = img.shape
	#print(imgInfo)
	height = imgInfo[0]
	width = imgInfo[1]
	channel = imgInfo[2]
	a = np.random.randint(50,100,size= [height,width,channel])
	dstImg = np.zeros((height,width,channel),np.uint8)
	dstImg = img + a
	cv2.imwrite(output,dstImg)
	k = cv2.waitKey(0)






