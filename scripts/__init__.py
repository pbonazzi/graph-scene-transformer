import torch
import os
import yaml
from easydict import EasyDict as edict
from render.colors import bcolors
from network.models.model import BubbleBee
import numpy as np

def euclidian_distance(vec1, vec2):
    return  torch.sqrt((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2 +(vec1[2]-vec2[2])**2)


def load_config(path):
    with open(path, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    return cfg


def select_device():
    if torch.cuda.is_available():
        print(f"{bcolors.OKGREEN} GPU available {bcolors.ENDC}")
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        print('GPU description :', torch.cuda.get_device_name(
            0), device_id)
        device = torch.device("cuda")
    else:
        print(
            f"{bcolors.OKGREEN} GPU not available, using CPU{bcolors.ENDC}")
        device = torch.device("cpu")
    return device

def total_param(params: dict) -> int:
    model = BubbleBee(params)
    counter = 0
    for param in model.parameters():
        counter += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', counter)
    return counter

def get_mean_and_std(graph, feature):

    assert feature in ["raw_location", "raw_dimension"]

    f = [g.ndata[feature] for g in graph]
    f = torch.cat(f, dim=0)

    f_x_mean, f_x_std = torch.mean(
        f[:, :1]), torch.std(f[:, :1])
    f_y_mean, f_y_std = torch.mean(
        f[:, 1:2]), torch.std(f[:, 1:2])
    f_z_mean, f_z_std = torch.mean(
        f[:, 2:3]), torch.std(f[:, 2:3])

    f_mean = [f_x_mean.item(
    ), f_y_mean.item(), f_z_mean.item()]

    f_std = [f_x_std.item(
    ), f_y_std.item(), f_z_std.item()]

    return f_mean, f_std
