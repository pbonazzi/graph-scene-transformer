from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
import os
import pickle
import os
import sys
import inspect
import time
import warnings
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from render.three_d import get_object_mesh, get_texturemesh, get_lineset
from render.utils import compute_box_3d
from data.utils import get_labels
from data.data.threed_ssg.constructor import Dataset3DSSG

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

    filename = os.path.join("~", "Desktop", "ATISS", "output", "threed_ssg", "threed_ssg.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    scene_graph = 0

    graph = data[scene_graph][0]
    id = data[scene_graph][1]

    path = os.path.join("test", "visualization", time.strftime('%y_%m_%d_%Hh%Mm'))
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path)

    labels = get_labels(graph,  "threed_ssg", "nodes")
    print(graph.ndata["global_id"])
    location = graph.ndata["raw_location"]
    dimension = graph.ndata["raw_dimension"]
    orientation = graph.ndata["orientation"]

    if scene_graph == 0:
        mesh = get_texturemesh(scan_id=id)
        mesh.compute_vertex_normals()
        mesh.create_coordinate_frame(size=1, origin=(0, 0, 0))
        writer.add_3d("scan", to_dict_batch([mesh]), step=0)

    for i in range(graph.number_of_nodes()):
        points = compute_box_3d(dimension[i], location[i], orientation[i])
        lines = get_lineset(points=points, rgb=[0, 0, 1])  # blue is score
        writer.add_3d(str(i) + "box" + labels[i], to_dict_batch([lines]), step=0)
        try:
            print("Loading label.. : ", labels[i])
            mesh = get_object_mesh(label=labels[i], color=[0.5, 0.5, 0.5],
                                   dimension=dimension[i], location=location[i],
                                   angle=orientation[i])
            writer.add_3d(str(i) + labels[i], to_dict_batch([mesh]), step=0)
        except IndexError:
            print("Problems with label : ", labels[i])

    writer.close()
