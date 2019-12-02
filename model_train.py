import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataloader
import net
import ssim.pytorch_ssim


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_clearimg', type=str, default="/home/deepcrying_dl/11785-DL-Project/data/train/")
    parser.add_argument('--path_hazyimg', type=str, default="/home/deepcrying_dl/11785-DL-Project/data/output/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    config = parser.parse_args()
    return config


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dehaze_net = net.AODNet().to(device)
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.DataLoader(
        config.path_clearimg, config.path_hazyimg)
    val_dataset = dataloader.DataLoader(
        config.path_clearimg, config.path_hazyimg, mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    # criterion = nn.MSELoss().to(device)
    criterion = ssim.pytorch_ssim.SSIM(window_size=11)
    optimizer = torch.optim.Adam(dehaze_net.parameters(
    ), lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(dehaze_net.parameters(
    # ), lr=config.lr, weight_decay=config.weight_decay)
    dehaze_net.train()
    for epoch in range(config.num_epochs):
        for iteration, (clear_img, hazy_img) in enumerate(train_loader):
            clear_img = clear_img.to(device)
            hazy_img = hazy_img.to(device)
            clean_image = dehaze_net(hazy_img)
            loss = criterion(clean_image, clear_img)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(
                dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Epoch", epoch + 1, ": Loss at iteration", iteration + 1, ":", loss.item())

        # Validation Stage
        for iter_val, (clear_img, hazy_img) in enumerate(val_loader):
            clear_img = clear_img.to(device)
            hazy_img = hazy_img.to(device)
            clean_image = dehaze_net(hazy_img)
            torchvision.utils.save_image(torch.cat(
                (hazy_img, clean_image, clear_img), 0), config.sample_output_folder + str(iter_val + 1) + ".jpg")
        torch.save(dehaze_net.state_dict(),
                   config.snapshots_folder + "Epoch" + str(epoch + 1) + '.pt')


if __name__ == "__main__":

    config = init_parser()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
