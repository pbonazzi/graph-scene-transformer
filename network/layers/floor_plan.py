from PIL import Image
import os
from torchvision import transforms
import torch
from collections import Callable
from config.paths import THREED_SSG_PLUS

def layout_encoder(ids: str, resnet50: Callable, device: str, bounding_box:bool, split:bool=False):
    """ From the scan ids , retrieve the floor plan and embed it with resnet50

        Parameters
        ----------
        - id: the scan id
        - resnet50 [nn.Embedding] : embedding of ResNet50
        - device

        Return
        ----------
        - enc_fp [list] : embedding of the floorplan

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d

    """
    input_images = []
    for id in ids:
        id = str(id)
        if split:
            id = id[:id.index("_")]

        if bounding_box:
            filename = os.path.join(THREED_SSG_PLUS, "floorplans",  "results", "bounding_boxes", "largest_inside", id+'.jpg')
        else:
            filename = os.path.join(THREED_SSG_PLUS, "floorplans", "results", "pointcloud2d", id+'.jpg')

        input_image = Image.open(filename)

        # crop and normalize the binary image to torch.Size([3, 224, 224])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).to(device)
        input_images.append(input_tensor)

    with torch.no_grad():
        enc_fp = resnet50(torch.stack(input_images))

    return enc_fp
