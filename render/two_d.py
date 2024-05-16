""" Utils to render bounding boxes in Matplotlib"""

# import libraries
from render.three_d import get_shapenet_labels
from data.utils import get_labels, get_vocabulary
from config.paths import THREED_SSG_PLUS
import networkx as nx
import torch
import dgl
import cv2
import matplotlib.pyplot as plt
import os.path
import json
import open3d as o3d
import numpy as np
import matplotlib

matplotlib.use('Agg')

# import custom methods


def get_scene_graph_diagram(graph: dgl.DGLGraph, filename: str = "", random_layout: bool = False):
    nx_graph, graph, edge_labels, _ = process_graph(graph)

    plt.interactive(False)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    if not random_layout:
        pos = {}
        location = graph.ndata["raw_location"]
        for node in range(graph.number_of_nodes()):
            pos[node] = np.array(location[node][:2])
    else:
        pos = nx.random_layout(nx_graph)

    category = graph.ndata["global_id"]
    nx.draw_networkx_nodes(nx_graph, pos, node_color=[
                           [i / 529] for i in category], node_size=100, alpha=1)
    nx.draw_networkx_edge_labels(
        nx_graph, pos, edge_labels=edge_labels, font_size=8)
    ax = plt.gca()
    for i, e in enumerate(nx_graph.edges):
        ax.annotate("",
                    xy=pos[e[1]], xycoords='data',
                    xytext=pos[e[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2]))))

    labels_nodes = get_labels(graph, "nodes")
    for i, e in enumerate(nx_graph.nodes):
        ax.annotate(labels_nodes[i],
                    xy=pos[i], xycoords='data',
                    xytext=pos[i] + 0.01, textcoords='data',
                    fontsize=12
                    )

    plt.rcParams.update({"figure.figsize": (50, 10)})
    plt.axis('off')
    if len(filename) > 0:
        plt.savefig(filename)
        plt.close()
        return

    return fig


def process_graph(graph: dgl.DGLGraph):
    graph = graph.clone()
    num_nodes = graph.number_of_nodes()
    
    check = graph.edata["feat"]
    num_edges = graph.number_of_edges()
    for i in range(num_edges - 1, -1, -1):
        if check[i] in [27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40]:
            edge_pair = torch.tensor([i])
            graph.remove_edges(edge_pair)
    nx_graph = dgl.to_networkx(
        graph, node_attrs=["global_id"], edge_attrs=["feat"])
    labels_nodes = get_labels(graph, "nodes")
    labels_edges = get_labels(graph, "edges")

    pairs = []
    triplet_labels = []
    edge_vocabulary = get_vocabulary(os.path.join(THREED_SSG_PLUS, "vocab", "relationships.tsv"))
    for i, (n1, n2, n3) in enumerate(nx_graph.edges(data=True)):
        edge_id = n3.get('feat').item()
        pairs.append(((n1, n2), edge_vocabulary.get(edge_id)))

        src_label = labels_nodes[n1]
        dst_label = labels_nodes[n2]
        triplet_labels.append(
            ((n1, n2), [src_label, dst_label, edge_vocabulary.get(edge_id)]))
    edge_labels = dict(pairs)

    return nx_graph, graph, edge_labels, triplet_labels


def attention_heatmap(attention_scores: np.ndarray, categories: list):
    plt.interactive(False)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    im = ax.imshow(attention_scores)

    # Create color bar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.15)
    cbar.ax.set_ylabel("Attention scores", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, rotation=45, ha="right")

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # ax.set_title(title)
    fig.tight_layout()

    return fig


def get_average_attention_map(graph: dgl.DGLGraph, params: dict, from_: str, of_: str, labels: list):
    if from_ == "edges":
        attention_score = np.zeros(
            (graph.number_of_edges(), graph.number_of_edges()))
        for i in range(params["n_of_attention_heads"]):
            att = graph.edata[of_ + "_score_of_head_" +
                              str(i)].cpu().detach().numpy().squeeze()
            attention_score = attention_score + att
        attention_scores = attention_score / params["n_of_attention_heads"]
    else:
        if params["floor_plan"]:
            attention_score = np.zeros(
                (graph.number_of_nodes(), graph.number_of_nodes()+1))
        else:
            attention_score = np.zeros(
                (graph.number_of_nodes(), graph.number_of_nodes()))

        for i in range(params["n_of_attention_heads"]):
            att = graph.ndata[of_ + "_score_of_head_" +
                              str(i)].cpu().detach().numpy().squeeze()
            attention_score = attention_score + att
        attention_scores = attention_score / params["n_of_attention_heads"]

    fig = attention_heatmap(
        attention_scores=attention_scores, categories=labels)

    return fig


def img_points(xyz: np.array):
    x, y = [i[0] for i in xyz], [i[1] for i in xyz]
    # min_x, min_y = abs(min(x)), abs(min(y))
    # max_x, max_y = max(x)+min_x, max(y)+min_y
    max_x, max_y = max(x), max(y)

    maxi = max(max_x, max_y)
    scal = int(520 / maxi)
    xy = np.zeros((len(xyz), 2), dtype=int)
    for i in range(len(xyz)):
        # xy[i][0]=int((xyz[i][0]+min_x)*scal)
        # xy[i][1]=int((xyz[i][1]+min_y)*scal)
        xy[i][0] = int(xyz[i][0] * scal)
        xy[i][1] = int(xyz[i][1] * scal)

    return xy


def plot_img_points(xy: np.array):
    """ Save the plots

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(11, 11))  # 1 in	96 pixel (X)

    axs[0].plot(xy[:, 0], xy[:, 1], ',')  # pixel marker
    axs[1].plot(xy[:, 0], xy[:, 1], 's')
    axs[0].set_aspect('equal', adjustable='box')
    axs[1].set_aspect('equal', adjustable='box')
    axs[0].axis('off')
    axs[1].axis('off')

    return fig


def save_plot(xy: np.array, filename: str):
    """ Save the plot

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    plt.figure(figsize=(5.5, 5.5), dpi=80)  # 1 in 	96 pixel (X)
    plt.plot(xy[:, 0], xy[:, 1], 's', color="black")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(filename)


def plot_2d_bbox(corners_by_scan_id_dic: dict, floor_id_dict: dict, wall_id_dict: dict, scan_id: str, xy=None):
    """ Plot the bounding boxes in 2D

    Author:
    https://github.com/uzh-rpg/scene_graph_3d
    """
    scan = corners_by_scan_id_dic.get(scan_id)
    plt.figure(figsize=(10, 20))
    for obj_key, obj_val in scan.items():
        if obj_key in floor_id_dict.get(scan_id):
            plot_arg = 'bo-'
            plot_id = True
        elif obj_key in wall_id_dict.get(scan_id):
            plot_arg = 'go-'
            plot_id = True
        else:
            plot_arg = 'ro-'
            plot_id = False

        for i in range(len(obj_val)):
            if i < 3:
                plt.plot([obj_val[i][0] * 30, obj_val[i + 1][0] * 30], [obj_val[i][1] * 30, obj_val[i + 1][1] * 30],
                         plot_arg)
            else:  # connect last point with 1st point
                plt.plot([obj_val[i][0] * 30, obj_val[0][0] * 30],
                         [obj_val[i][1] * 30, obj_val[0][1] * 30], plot_arg)
        if plot_id:
            plt.text(obj_val[0][0] * 30, obj_val[0][1] * 30, obj_key)

    if xy is not None:
        plt.plot(xy[:, 0], xy[:, 1], ',')
    plt.axis('equal')
    plt.show()


def cp_visualize_2d(xyz):
    """ Visualize point cloud in 2D.
    Used to study different methods on the floor plan computing.

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    for i in range(len(xyz)):
        xyz[i][-1] = 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


def compute_morph_transf(img: np.array, scan_id: str):
    """ Morphological transformation of an image

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    root = "morph_img" + scan_id

    kernel = np.ones((30, 30), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(os.path.join(root, 'erosion.jpg'), erosion)

    dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite(os.path.join(root, 'dilation.jpg'), dilation)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(root, 'opening.jpg'), opening)
