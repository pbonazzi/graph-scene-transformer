# import libraries
from cv2 import threshold
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import dgl
from geomloss import SamplesLoss
import json
from network.metrics.iou import box3d_iou, compute_iou_box3d

# import custom methods
from render.utils import compute_box_3d


def normalized_sum(v1, v2):
    if (len(v1) + len(v2)) == 0:
        return 1.0
    return (sum(v1) + sum(v2)) / (len(v1) + len(v2))


def get_diversity(scores: list, params: dict):
    # denormalize features
    concatenated_scores = np.concatenate(scores, 0)
    scores_size = concatenated_scores[:, :3]
    scores_loc = concatenated_scores[:, 3:6]
    scores_ori = np.argmax(
        concatenated_scores[:, 6:], axis=1) * 90 / params["n_of_orientation_bins"]

    # get diversity
    n_of_objects = scores[0].shape[0]
    n_of_inference_runs = int(concatenated_scores.shape[0] / n_of_objects)
    diversity_size, diversity_loc, diversity_ori = 0, 0, 0
    for i in range(n_of_objects):
        indexes = np.arange(
            i, n_of_objects * n_of_inference_runs, n_of_inference_runs)
        diversity_size += (np.std(scores_size[indexes, :1]) + np.std(
            scores_size[indexes, 1:2]) + np.std(scores_size[indexes, 2:3])) / 3
        diversity_loc += (np.std(scores_loc[indexes, :1]) + np.std(
            scores_loc[indexes, 1:2]) + np.std(scores_loc[indexes, 2:3])) / 3
        diversity_ori += np.std(scores_ori[indexes])

    return diversity_size / n_of_objects, diversity_loc / n_of_objects, diversity_ori / n_of_objects


def sinkhorn_wassersein(graph: dgl.DGLGraph, scores: torch.tensor, params: dict):
    # denormalize features
    graph = graph.local_var()
    scores_dim = scores[:, :3]
    targets_dim = graph.ndata["raw_dimension"].cpu().detach()

    scores_loc = scores[:, 3:6]
    targets_loc = graph.ndata["raw_location"].cpu().detach()

    points_x, points_y = [], []
    num_obj = graph.number_of_nodes()

    for i in range(num_obj):
        points_x.append(compute_box_3d(
            dim=scores_dim[i], location=scores_loc[i]))
        points_y.append(compute_box_3d(
            dim=targets_dim[i], location=targets_loc[i]))

    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    return loss(torch.tensor(points_x).reshape(num_obj * 8, 3), torch.tensor(points_y).reshape(num_obj * 8, 3))


def constraints_validation(accuracy: dict, g: dgl.DGLGraph, scores: torch.tensor, params: dict):
    """

    Args:
        accuracy: a dictionary of relationship classes metrics
        g: the batch graph containing the edge features we want to validate
        scores: a matrix containing the estimated dimensions and locations
        params: the parameters used during training

    Returns: the accuracy dictionary with the results of the constraint verification process

    """
    # overlapping threshold
    thr = 0.5

    # triplets
    g = g.local_var()
    src_nodes = g.edges()[0]
    dst_nodes = g.edges()[1]
    feat = g.edata["feat"]

    # de normalize features
    scores_dim = scores[:, :3]
    targets_dim = g.ndata["raw_dimension"]

    scores_loc = scores[:, 3:6]
    targets_loc = g.ndata["raw_location"]

    scores_ori = np.argmax(scores[:, 6:], axis=1) * \
        90 / params["n_of_orientation_bins"]

    est = 0

    for i in range(len(feat)):
        # predictions
        src_node_location = scores_loc[src_nodes[i]]
        dst_node_location = scores_loc[dst_nodes[i]]
        src_node_dimension = scores_dim[src_nodes[i]]
        dst_node_dimension = scores_dim[dst_nodes[i]]

        src_ori = scores_ori[src_nodes[i]]
        dst_ori = scores_ori[dst_nodes[i]]

        vol_src = src_node_dimension[0] * \
            src_node_dimension[1] * src_node_dimension[2]
        vol_dst = dst_node_dimension[0] * \
            dst_node_dimension[1] * dst_node_dimension[2]

        boxsrc = compute_iou_box3d(dim=src_node_dimension, location=src_node_location,
                                   angle=src_ori)
        boxdst = compute_iou_box3d(dim=dst_node_dimension, location=dst_node_location,
                                   angle=dst_ori)

        inter = box3d_iou(boxsrc, boxdst)
        # targets
        # src_node_location_targets = targets_loc[src_nodes[i]]
        # dst_node_location_targets = targets_loc[dst_nodes[i]]
        # src_node_dimension_targets = targets_dim[src_nodes[i]]
        # dst_node_dimension_targets = targets_dim[dst_nodes[i]]
        # vol_src_targets = src_node_dimension_targets[0] * \
        #     src_node_dimension_targets[1] * src_node_dimension_targets[2]
        # vol_dst_targets = dst_node_dimension_targets[0] * \
        #     dst_node_dimension_targets[1] * dst_node_dimension_targets[2]

        if feat[i] == 2:
            # # left
            # if src_node_location_targets[0] < dst_node_location_targets[0]:
            if src_node_location[0] > dst_node_location[0] or thr < inter:
                accuracy['left'].append(0)
            else:
                accuracy['left'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 3:
            # # right
            # if src_node_location_targets[0] > dst_node_location_targets[0]:
            if src_node_location[0] < dst_node_location[0] or thr < inter:
                accuracy['right'].append(0)
            else:
                accuracy['right'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 4:
            # # front
            # if src_node_location_targets[1] < dst_node_location_targets[1]:
            if src_node_location[1] > dst_node_location[1] or thr < inter:
                accuracy['front'].append(0)
            else:
                accuracy['front'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 5:
            # # behind
            # if src_node_location_targets[1] > dst_node_location_targets[1]:
            if src_node_location[1] < dst_node_location[1] or thr < inter:
                accuracy['behind'].append(0)
            else:
                accuracy['behind'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 8:
            # # bigger than
            # if vol_src_targets > vol_dst_targets:
            if vol_src < vol_dst:
                accuracy['bigger'].append(0)
            else:
                accuracy['bigger'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 9:
            # # smaller than
            # if vol_src_targets < vol_dst_targets:
            if vol_src > vol_dst:
                accuracy['smaller'].append(0)
            else:
                accuracy['smaller'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 10:
            # # higher
            # if src_node_location_targets[2] > dst_node_location_targets[2]:
            if src_node_location[2]/2 + src_node_dimension[2] < dst_node_location[2] / 2 + dst_node_dimension[2]:
                accuracy['higher'].append(0)
            else:
                accuracy['higher'].append(1)
            # else:
            #     est += 1

        elif feat[i] == 11:
            # # lower
            # if src_node_location_targets[2] < dst_node_location_targets[2]:
            if src_node_location[2]/2 + src_node_dimension[2] > dst_node_location[2] / 2 + dst_node_dimension[2]:
                accuracy['lower'].append(0)
            else:
                accuracy['lower'].append(1)
            # else:
            #     est += 1

    # if est > 0:
    #     print("Annotations errors : ", est)

    return accuracy


def accuracy(scores, targets):
    """ This function is not used at the moment

    Author : Vijay Dwivedi
    https://github.com/graphdeeplearning/graphtransformer

    Modified by : Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d

    """

    S = np.argmax(torch.nn.Softmax(dim=1)(
        targets).cpu().to(int).numpy(), axis=1)
    C = np.argmax(torch.nn.Softmax(dim=1)(
        scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = np.argmax(torch.nn.Softmax(dim=1)(
        targets).cpu().to(int).numpy(), axis=1)
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc
