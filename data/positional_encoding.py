# import libraries
import hashlib
import dgl
import numpy as np
from scipy import sparse as sp
import torch


def laplacian_positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> dgl.DGLGraph:
    """ Graph positional encoding v/ Laplacian eigenvectors

    Author : Vijay Dwivedi
    https://github.com/graphdeeplearning/graphtransformer

    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr")
    D = sp.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - D * A * D

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # pad 0 to eigenvectors at end if number of nodes less than pos_enc_dim
    if EigVec.shape[1] < pos_enc_dim + 1:
        pad = pos_enc_dim + 1 - EigVec.shape[1]
        pad_EigVec = np.pad(EigVec, ((0, 0), (0, pad)), constant_values=0)
        EigVec = pad_EigVec

    g.ndata['lap_pos_enc'] = torch.from_numpy(
        EigVec[:, 1:pos_enc_dim + 1]).float()

    return g


def wl_positional_encoding(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """ WL-based absolute positional embedding adapted from

        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert

    """

    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + \
                sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v,
                            k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g
