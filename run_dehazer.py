import torch
import torchvision
import torch.optim
import numpy as np
from PIL import Image
import glob
import dataloader
import net


def dehaze(hazyimg_path, clearimg_path):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    data_hazy = Image.open(hazyimg_path).convert('RGB')
    data_hazy = (np.asarray(data_hazy) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.to(device).unsqueeze(0)

    data_clear = Image.open(clearimg_path).convert('RGB')
    data_clear = (np.asarray(data_clear) / 255.0)
    data_clear = torch.from_numpy(data_clear).float()
    data_clear = data_clear.permute(2, 0, 1)
    data_clear = data_clear.to(device).unsqueeze(0)

    dehaze_net = net.AODNet().to(device)
    dehaze_net.load_state_dict(torch.load('snapshots/dehaze_model_with_dcp.pt'))

    clean_image = dehaze_net(data_hazy)
    torchvision.utils.save_image(torch.cat(
        (data_hazy, clean_image, data_clear), 0), "results/" + hazyimg_path.split("/")[-1])


if __name__ == '__main__':
    hazy_list = glob.glob("sample_dataset/hazedimg/*")
    clear_list = glob.glob("sample_dataset/clearimg/*")
    hazy_list.sort()
    clear_list.sort()

    for hazy_image, clear_image in zip(hazy_list, clear_list):
        dehaze(hazy_image, clear_image)
        print(hazy_image, "done!")
