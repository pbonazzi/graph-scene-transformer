# import libraries
import torch
import torch.utils.data
import dgl
import numpy as np
import networkx as nx
import csv
import os
import random
from collections.abc import Callable
from typing import Any
import json

# import custom methods
from data.normalization import Normalizer
from config.paths import THREED_SSG_PLUS

def transform_graphs(func: Callable, data, arg: Any = None):
    if arg is None:
        return [func(g) for g in data]
    else:
        return [func(g, arg) for g in data]


def get_vocabulary(path: str, load_colors: bool = False):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv)  # skip first row
    if load_colors:
        return {int(rows[0]): [float(rows[2]), float(rows[3]), float(rows[4])] for rows in read_tsv}
    vocabulary = {int(rows[0]): rows[1] for rows in read_tsv}
    return vocabulary


def get_labels(graph: dgl.DGLGraph, labels_type: str):

    labels = []
    if labels_type == "nodes":
        vocabulary = get_vocabulary(os.path.join(THREED_SSG_PLUS, "vocab", "objects.tsv"))
        category = graph.ndata["global_id"]
        for nodes in category:
            labels.append(vocabulary.get(nodes.item()))
            
    elif labels_type == "nodes_colors":
        vocabulary = get_vocabulary(os.path.join(THREED_SSG_PLUS, "vocab", "objects.tsv"), "colors")
        category = graph.ndata["global_id"]
        for nodes in category:
            labels.append(vocabulary.get(nodes.item()))
    else:
        vocabulary = get_vocabulary(os.path.join(
            THREED_SSG_PLUS, "vocab", "relationships.tsv"))
        category = graph.edata["feat"]
        for edges in category:
            labels.append(vocabulary.get(edges.item()))
    return labels


def get_orientation(graph, device, ohc: bool = False, num_of_bins: int = 24):
    orientation = graph.ndata['orientation'].to(device)
    orientation = torch.div(orientation, 90 / num_of_bins,
                            rounding_mode="floor").to(device, dtype=torch.int64)
    if not ohc:
        return orientation
    return torch.nn.functional.one_hot(orientation, num_of_bins).to(device)


def get_category(graph, device, ohc: bool = False, num_of_objects: int = 529):
    category = graph.ndata['global_id']
    if not ohc:
        return graph.ndata['global_id'].to(device)
    else:
        return torch.nn.functional.one_hot(category, num_of_objects).to(device)


def get_location(graph: dgl.DGLGraph, normalized: bool = True, decomposed: bool = True):
    if normalized:
        location = graph.ndata['location']
    else:
        location = graph.ndata['raw_location']
    if not decomposed:
        return location
    location_x = location[:, 0].unsqueeze(1)
    location_y = location[:, 1].unsqueeze(1)
    location_z = location[:, 2].unsqueeze(1)
    return location_x, location_y, location_z


def get_dimension(graph: dgl.DGLGraph, normalized: bool = True, decomposed: bool = True):
    if normalized:
        dimension = graph.ndata['dimension']
    else:
        dimension = graph.ndata['raw_dimension']
    if not decomposed:
        return dimension
    dimension_l = dimension[:, 0].unsqueeze(1)
    dimension_w = dimension[:, 1].unsqueeze(1)
    dimension_h = dimension[:, 2].unsqueeze(1)
    return dimension_l, dimension_w, dimension_h


def search_index_of_scan_id(data, scan_id: str) -> int:
    for index in range(len(data)):
        candidate = data.__getitem__(index)[1]
        if candidate == scan_id:
            return index


def get_indexes(len_of_data: int, ratio: list = None, idx_spec: int = 0):
    if ratio is None:
        ratio = [0.8, 0.2]
    data_idx = np.arange(len_of_data)
    data_idx = np.delete(data_idx, idx_spec)
    random.Random(4).shuffle(data_idx)

    train_slice = int(len_of_data * ratio[0])
    val_slice = train_slice + int(len_of_data * ratio[1])

    idx_train = np.concatenate((data_idx[:train_slice], np.array([idx_spec])))
    idx_val = data_idx[train_slice:val_slice]

    return idx_train, idx_val


def partition_data(data_idx: list, data):
    part_data = []
    for i in data_idx:
        try:
            part_data.append((data.__getitem__(i)[0], data.__getitem__(i)[1]))
        except IndexError:
            continue
    # part_data = [(data.__getitem__(i)[0], data.__getitem__(i)[1]) for i in data_idx]

    return part_data


def write_scan_ids(root: str, save_path: str):
    # load json files
    relationships_train_path = os.path.join(root, "relationships_train.json")
    relationships_val_path = os.path.join(
        root, "relationships_validation.json")
    rel_train = json.load(open(relationships_train_path, ))
    rel_val = json.load(open(relationships_val_path, ))
    
    # get scan ids
    train_scan_ids = [g['scan'] for g in rel_train['scans']]
    val_scan_ids = [g['scan'] for g in rel_val['scans']]

    # write into txt files
    train_ids = open(os.path.join(save_path, "train_ids.txt"), "w")
    for train_id in set(train_scan_ids):
        train_ids.write(train_id + "\n")
    train_ids.close()

    val_ids = open(os.path.join(save_path, "val_ids.txt"), "w")
    for val_id in set(val_scan_ids):
        val_ids.write(val_id + "\n")
    val_ids.close()

    return


def partition_by_ids(id_list: list, data):
    part_data = []
    for i in range(len(data)):
        g = data.__getitem__(i)[0]
        scan_id = data.__getitem__(i)[1]
        if scan_id in id_list:
            part_data.append((g, scan_id))

    return part_data


def normalize_graph(g: dgl.DGLGraph, N: Normalizer) -> dgl.DGLGraph:
    raw_location = g.ndata["raw_location"].clone()
    g.ndata["location"] = N.normalize(
        data=raw_location, feature="location")

    raw_dimension = g.ndata["raw_dimension"].clone()
    g.ndata["dimension"] = N.normalize(
        data=raw_dimension, feature="dimension")

    return g


def make_full_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['global_id'] = g.ndata['global_id']
    full_g.ndata['location'] = g.ndata['location']
    full_g.ndata['dimension'] = g.ndata['dimension']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()

    try:
        full_g.edata['feat'] = g.edata['feat']
    except KeyError:
        pass

    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except KeyError:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except KeyError:
        pass

    return full_g


def get_max_graph_size(dataset):
    max_graph_size = 0
    for graph, _ in dataset:
        max_graph_size = max(max_graph_size, graph.number_of_nodes())
    return max_graph_size
