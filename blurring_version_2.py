import matplotlib.pyplot as plt
import os

mosaic_size = 10                     #need to change
path = 'mosaic_10_part'              #need to change
files = os.listdir(path)
imageName = []
for file in files:
    if not os.path.isdir(file):
        imageName.append(path+"/"+file)

print(len(imageName))
for i in range(len(imageName)): 
    #output need to change
    output = "mosaic_10_part/part"+"_"+str(mosaic_size)+"_"+ imageName[i].split('/')[1]
    street = plt.imread(imageName[i])
    mosaic_part = street[40:620, 400:900]
    #mosic picture
    after_mosaic_part = mosaic_part[::mosaic_size,::mosaic_size]
    #plt.imshow(after_mosaic_part)
    x_axis = after_mosaic_part.shape[0]
    y_axis = after_mosaic_part.shape[1]
    street_m = street.copy()
    for i in range(x_axis):
        for j in range(y_axis):
            street_m[40+i*mosaic_size:70+i*mosaic_size, 400+j*mosaic_size: 430 +j*mosaic_size] = after_mosaic_part[i,j]
    plt.imshow(street_m)
    #remove white space around picture and save picture
    fig, ax = plt.subplots()
    im = street_m[:,:,(0,1,2)]
    ax.imshow(im,aspect='equal')
    plt.axis('off')
    height,width,channels=im.shape
    fig.set_size_inches(width/100.0/3.0,height/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(output,dpi = 300)
    #plt.show()
