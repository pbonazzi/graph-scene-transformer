from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
import pandas as pd
import os
import sys
import inspect
import warnings

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from render.three_d import get_object_mesh
from data.utils import get_vocabulary

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))
    root = os.path.join("render", "data", "shapenet_sem")
    df = pd.read_csv(os.path.join(root, "metadata.csv"))
    labels_dict = get_vocabulary("input/data/threed_ssg/raw/vocab/objects.tsv")

    for i in range(len(labels_dict.items())):
        # candidates = df[df.category == labels[i][1]]
        candidates = df[df.wnlemmas == list(labels_dict.values())[i]]
        if len(candidates) > 0:
            if candidates:
                save.append(labels_dict.values()[i])
        else:
            print("Problems with label : ", list(labels_dict.values())[i])

    print("Found: ", len(save))
    print(save)
