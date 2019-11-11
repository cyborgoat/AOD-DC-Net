import torch
import torchvision
import torch.optim
import numpy as np
from PIL import Image
import glob
import dataloader
import net


def dehaze(hazyimg_path):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    data_hazy = Image.open(hazyimg_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.to(device).unsqueeze(0)

    dehaze_net = net.AODNet().to(device)
    dehaze_net.load_state_dict(torch.load('snapshots/dehaze_model.pt'))

    clean_image = dehaze_net(data_hazy)
    torchvision.utils.save_image(torch.cat(
        (data_hazy, clean_image), 0), "results/" + hazyimg_path.split("/")[-1])


if __name__ == '__main__':
    test_list = glob.glob("sample_dataset/hazedimg/*")

    for image in test_list:
        dehaze(image)
        print(image, "done!")
