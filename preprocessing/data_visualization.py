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
from torch.utils.data.sampler import SubsetRandomSampler

train_path = sys.argv[1]
test_path = sys.argv[2]
action = sys.argv[3]

BATCH_SIZE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_stat(dataset):
    num_samples = len(dataset.samples)
    num_classes = len(dataset.classes)

    return num_samples, num_classes


def visualize_img(dataloader):
    for batch_num, (feats, labels) in enumerate(dataloader):
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        print(feats.shape)
        for feat in feats:
            feat = torch.squeeze(feat)
            pilTrans = transforms.ToPILImage()
            pilImg = pilTrans(feat)
            plt.imshow(pilImg)
            plt.show()


def visualize_stat(train_sample, validation_samples):
    objects = ('Train_Set', 'Vlidation_set')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [train_sample, validation_samples], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('# of samples')
    plt.title('Data Distribution')

    plt.show()


def data_splitting(dataset):
    batch_size = 16
    split_distribution = [0.8, 0.2]
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_train = int(np.floor(split_distribution[0] * dataset_size))
    split_validation = int(np.floor(split_distribution[1] * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split_train], \
                                               indices[split_train:]
    print("Train indices:{}\nValidation indices:{}".format(train_indices, val_indices))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    num_samples = [len(train_sampler), len(valid_sampler)]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader,num_samples


if __name__ == "__main__":
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=torchvision.transforms.ToTensor())
    # num_samples, num_classes = data_stat(train_dataset)
    train_loader, validation_loader,num_samples = data_splitting(train_dataset)
    if action == "img":
        visualize_img(train_loader)
    elif action == "stat":
        visualize_stat(num_samples[0], num_samples[1])
