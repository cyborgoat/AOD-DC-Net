import datetime
import numpy as np
import torch
import torchvision
import sys
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_path = sys.argv[1]
test_path = sys.argv[2]
dev_path = sys.argv[3]

BATCH_SIZE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_stat(dataset):
    num_samples = len(dataset.samples)
    num_classes = len(dataset.classes)

    return num_samples, num_classes


def visualize_img(dataloader):
    for batch_num, (feats, labels) in enumerate(dataloader):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        for feat in feats:
            feat = torch.squeeze(feat)
            pilTrans = transforms.ToPILImage()
            pilImg = pilTrans(feat)
            plt.imshow(pilImg)
            plt.show()
            break


def visualize_stat(train_sample, test_samples):
    objects = ('Train_Set', 'Test_Set')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [train_sample, test_samples], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('# of samples')
    plt.title('Train Dataset vs. Test Dataset')

    plt.show()


if __name__ == "__main__":
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    num_samples, num_classes = data_stat(train_dataset)
    visualize_img(train_dataloader)
    # visualize_stat(num_samples,num_samples)
