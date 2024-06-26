# import libraries
import numpy as np
import torch
from scipy.spatial import ConvexHull
import dgl
import json
import math
import scipy
import copy


def iou_boxes(graph: dgl.DGLGraph, scores: torch.tensor, params: json, Normalizer):
    """

    Args:
        graph: the batch graph containing the edge features we want to validate
        scores: a matrix containing the estimated dimensions and locations
        params: the parameters used during training

    Returns:
        Summed IOU3D of all boxes in the same scene, averaged across the number of batch graphs.

    """
    # de normalize features
    graph = graph.local_var()
    list_of_graphs = dgl.unbatch(graph)

    myscores = copy.deepcopy(scores)
    scores_dim = Normalizer.denormalize(myscores[:, :3], "dimension")
    scores_loc = Normalizer.denormalize(myscores[:, 3:6], "location")

    scores_ori = myscores[:, 6:params["n_of_orientation_bins"] + 6]

    batch_iou_scores, batch_iou_targets, start = 0, 0, 0

    for i in range(len(list_of_graphs)):
        targets_dim = list_of_graphs[i].ndata["raw_dimension"]
        targets_loc = list_of_graphs[i].ndata["raw_location"]
        targets_ori = list_of_graphs[i].ndata["orientation"]

        iou_scores, iou_gt = 0, 0
        num_obj = list_of_graphs[i].number_of_nodes()

        graph_scores_dim = scores_dim[start:start + num_obj, :]
        graph_scores_loc = scores_loc[start:start + num_obj, :]
        graph_scores_ori = scores_ori[start:start + num_obj, :]

        start += num_obj

        for j in range(num_obj):
            for k in range(j + 1, num_obj):
                box1_s = compute_iou_box3d(dim=graph_scores_dim[j], location=graph_scores_loc[j],
                                           angle=np.argmax(graph_scores_ori[j]).item()*3.75)
                box2_s = compute_iou_box3d(dim=graph_scores_dim[k], location=graph_scores_loc[k],
                                           angle=np.argmax(graph_scores_ori[k]).item()*3.75)

                box1_t = compute_iou_box3d(dim=targets_dim[j], location=targets_loc[j],
                                           angle=targets_ori[j].item())
                box2_t = compute_iou_box3d(dim=targets_dim[k], location=targets_loc[k],
                                           angle=targets_ori[k].item())

                iou_scores += box3d_iou(box1_s, box2_s)
                iou_gt += box3d_iou(box1_t, box2_t)
        batch_iou_scores += iou_scores / (num_obj*(num_obj-1)/2)
        batch_iou_targets += iou_gt / (num_obj*(num_obj-1)/2)

    return batch_iou_scores / len(list_of_graphs), batch_iou_targets / len(list_of_graphs)


def compute_iou_box3d(dim: list, location: list, angle: float = None):
    """ Compute 3D points coordinates from feature """

    l, w, h = dim[0], dim[1], dim[2]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2,   -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
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


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def box3d_iou(corners1, corners2):
    """ Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    """

    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]

    try:
        inter_area = convex_hull_intersection(rect1, rect2)[1]
        ymax = min(corners1[0, 1], corners2[0, 1])
        ymin = max(corners1[4, 1], corners2[4, 1])

        inter_vol = inter_area * max(0.0, ymax - ymin)

        vol1 = box3d_vol(corners1)
        vol2 = box3d_vol(corners2)

        volmin = min(vol1, vol2)

        iou = inter_vol / volmin  # (vol1 + vol2 - inter_vol)

        return iou
    except scipy.spatial.qhull.QhullError:
        return 1
