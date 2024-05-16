""" Utils to render bounding boxes in Tensorboard"""

# import libraries
import numpy as np
import pandas as pd
import os
import math
import dgl
import json


# import custom methods
from data.utils import get_orientation, get_labels, get_vocabulary
from data.normalization import Normalizer


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def compute_box_3d(dim: list, location: list, angle: float = None):
    """ Compute 3D points coordinates from feature
    Author:
    https://github.com/uzh-rpg/scene_graph_3d
    """
    # dim: 3
    # location: 3
    # angle: 1
    # return: 8 x 3

    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2]
    y_corners = [-w / 2, -w / 2, -w / 2, -w / 2, w / 2, w / 2, w / 2, w / 2]
    z_corners = [-h / 2, h / 2, -h / 2, h / 2, -h / 2, h / 2, -h / 2, h / 2]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

    if angle is not None:
        if angle > 90:
            angle = 90
        angle = math.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        corners = np.dot(R, corners)

    corners_3d = corners + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def edit_metadata(root: str = None):
    if root is None:
        root = os.path.join("render", "data", "shapenet_sem")
    df_raw = pd.read_csv(os.path.join(root, "metadata_sem.csv"))

    # select columns (10712 records)
    df = df_raw[['fullId', 'category', 'wnlemmas', 'up', 'front']]

    # select rows with valid up and front direction (10586 records)
    df['up'] = df['up'].fillna("0\,0\,1")
    df['front'] = df['front'].fillna("0\,1\,0")
    df[['up_x', 'up_y', 'up_z']] = df['up'].str.replace(
        "\\", "").str.split(',', expand=True).astype(float)
    df[['front_x', 'front_y', 'front_z']] = df['front'].str.replace(
        "\\", "").str.split(',', expand=True).astype(float)

    df_f = df[(abs(df['up_x']+df['up_y']+df['up_z']) == 1.0) &
              (abs(df['front_x']+df['front_y']+df['front_z']) == 1.0)]

    # combine labels
    df_f['labels'] = df_f[['category', 'wnlemmas']].apply(
        lambda x: ','.join(x[x.notnull()]), axis=1)

    # only keep items with labels in 3DSSG (6695 records)
    target_labels = list(get_vocabulary(
        os.path.join("input/data/threed_ssg/raw/vocab", "objects.tsv")).values())
    idx_list = []
    labels_list = []
    for index, row in enumerate(df_f.iterrows()):
        obj_set = set()
        for item in row[1]['labels'].split(','):
            if item.lower() in target_labels:
                obj_set.add(item.lower())
        if len(obj_set) > 0:
            labels_list.append(','.join(obj_set))
            idx_list.append(index)

    df_final = df_f.iloc[idx_list]
    df_final['label'] = labels_list

    # adjust for known wrong combined labels
    df_final['label'].replace('can,pot,toilet,stool,commode', 'toilet')

    df_final = df_final[['fullId', 'label', 'up_x',
                         'up_y', 'up_z', 'front_x', 'front_y', 'front_z']]

    df_final.to_csv(os.path.join(root, "metadata_3dssg.csv"), index=False)


class SceneInfoContainer:
    """ Class used to render a scene
    Author:
    https://github.com/uzh-rpg/scene_graph_3d
    """

    def __init__(self, graph: dgl.DGLGraph, params: dict, scores: np.ndarray, N: Normalizer, targets: bool = True):

        self.nodes = graph.number_of_nodes()
        self.norm_loc = params["normalize_location"]
        self.norm_dim = params["normalize_dimension"]

        self.labels = get_labels(
            graph=graph, dataset_name=params["dataset"], labels_type="nodes")
        self.oridim = params["n_of_orientation_bins"]

        if targets:
            self.directions = graph.ndata["direction"]

            # targets
            self.ori_targets = get_orientation(
                graph=graph, num_of_bins=self.oridim, device='cpu', ohc=True)
            self.loc_targets = graph.ndata["raw_location"].numpy()
            self.dim_targets = graph.ndata["raw_dimension"].numpy()

        # scores
        if N.normalize_location:
            loc_score = scores[:, 3:6]
            self.loc_scores = N.denormalize(data=loc_score, feature="location")
        else:
            self.loc_scores = scores[:, 3:6]

        if N.normalize_dimension:
            dim_score = scores[:, :3]
            self.dim_scores = N.denormalize(
                data=dim_score, feature="dimension")
            self.dim_scores = self.dim_scores.clip(min=1e-4)
            # for i, score in enumerate(self.dim_scores):
            #     self.dim_scores[i] = max(score, 1e-3)
            #assert all(i for i in score > 0), print(f'Target:{self.dim_targets[i, :3]}, Score:{ori_score[i, :3]}, Denormalization: {score}')
        else:
            self.dim_scores = scores[:, :3]

        self.ori_scores = scores[:, 6:params["n_of_orientation_bins"] + 6]
