from utils import generate_coordinate_system
from models.model import OWSLR
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import argparse
import yaml


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def semiLocal_xy(img_length):
    return  torch.meshgrid(torch.linspace(start=-img_length//2, end=img_length//2, steps=img_length),
                           torch.linspace(start=-img_length//2, end=img_length//2, steps=img_length),
                           indexing='ij')
    

def create_semiLocal_area(hr_cs, lr_unit_width, lr_unit_height, concentration_length, semiLocal_x, semiLocal_y):

        m = hr_cs[..., 0].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                           1,
                                                           concentration_length,
                                                           concentration_length)
        x = (m + semiLocal_x * lr_unit_width).clamp_(-1 + 1e-6, 1 - 1e-6)

        n = hr_cs[..., 1].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                           1,
                                                           concentration_length,
                                                           concentration_length)
        y = (n + semiLocal_y * lr_unit_height).clamp_(-1 + 1e-6, 1 - 1e-6)

        return torch.stack((x, y), dim=-1).squeeze(0), torch.stack(((m-x), (n-y)), dim=-1).squeeze(0)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--img_path')
    parser.add_argument('--upscale_factor', type=float)
    parser.add_argument('--model_path')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    batch_size = args.batch_size
    img_path = args.img_path
    upscale_factor = args.upscale_factor
    model_path = args.model_path

    model = OWSLR(**config["model"]["args"]).to(device)

    model.load_state_dict(torch.load(model_path)['model']['sd'])

    model.eval()


    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

    _, H, W = img.shape

    new_H = int(H * upscale_factor)
    new_W = int(W * upscale_factor)


    hr_training_coordinates = generate_coordinate_system((new_H, new_W),
                                                         represent_xy_format=True)


    idx = 0
    pred_hr_img = []

    coord = make_coord((new_H, new_W)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / new_H
    cell[:, 1] *= 2 / new_W


    t = config['data_norm']["inp"]
    inp_sub = torch.FloatTensor(t['sub']).view(1)
    inp_div = torch.FloatTensor(t['div']).view(1)

    x = model.make_feat(((img-inp_sub)/inp_div).unsqueeze(0).to(device))
    with torch.no_grad():

        while idx < hr_training_coordinates.shape[0]:
            check_size = min(idx + batch_size, hr_training_coordinates.shape[0])
            batch_hr_training_coordinates = hr_training_coordinates[idx : check_size]
            batch_cell = cell[idx : check_size]
            
            print(batch_hr_training_coordinates.shape)
            pred = model.find_rgb(x, batch_hr_training_coordinates.unsqueeze(0).to(device), batch_cell.unsqueeze(0)).to("cpu")

            pred_hr_img.append(pred.unsqueeze(0))


            idx = check_size

    pred_hr_img = torch.cat(pred_hr_img, dim=1)
    pred_hr_img = (pred_hr_img*inp_div + inp_sub).clamp(0, 1).view(new_H, new_W, 3).permute(2, 0, 1)

    transforms.ToPILImage()(pred_hr_img.to("cpu")).save(img_path.split(".")[0]+"x"+str(upscale_factor)+".jpg")


if __name__ == "__main__":
    test()

