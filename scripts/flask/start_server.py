from flask import Flask, request
import json
import warnings
import open3d as o3d
import os
import dgl
import torch
import pickle
import numpy as np

# import custom methods
from data.utils import get_vocabulary, get_labels
from data.positional_encoding import laplacian_positional_encoding
from scripts.flask.infer_graph import scene_generation
from scripts.bubblebee.train_bubble_bee import select_device
from render.three_d import get_shapenet_labels
from render.two_d import get_scene_graph_diagram
from data.dataset import Dataset3DSSG
# create the app
app = Flask(__name__)


def rgb_to_hex(r, g, b):
    return ('{:X}{:X}{:X}').format(int(r*255), int(g*255), int(b*255))


@app.route("/api/load_data", methods=["GET"])
def get_3dssg_cytoscape_elements():

    with open("input/data/threed_ssg/threed_ssg_subset_val.pkl", "rb") as f:
        data = pickle.load(f)
    elements = []
    shapenet_labels = get_shapenet_labels()

    for i in range(len(data)):
        scene_graph = []
        idx_nodes_not_shapenet = []
        graph = data[i][0]
        node_labels = get_labels(
            graph=graph, dataset_name="threed_ssg", labels_type="nodes")

        nodes_colors = get_labels(
            graph=graph, dataset_name="threed_ssg", labels_type="nodes_colors")
        for node in range(graph.number_of_nodes()):
            if node_labels[node] in shapenet_labels:
                scene_graph.append(
                    {"data": {"id": str(graph.ndata['id'][node].item()), "label": node_labels[node],
                              "color": "#"+str(rgb_to_hex(nodes_colors[node][0], nodes_colors[node][1], nodes_colors[node][2]))}})
            else:
                idx_nodes_not_shapenet.append(node)

        graph.remove_nodes(idx_nodes_not_shapenet)

        edge_labels = get_labels(
            graph=graph, dataset_name="threed_ssg", labels_type="edges")

        edges = graph.edges()
        for edge in range(graph.number_of_edges()):
            scene_graph.append({"data": {"source": str(graph.ndata['id'][edges[0][edge].item()].item(
            )), "target": str(graph.ndata['id'][edges[1][edge].item()].item()), "label": edge_labels[edge]}})
        elements.append(scene_graph)

    return {"graphs": elements}


@app.route("/api/get_points", methods=["POST", "GET"], strict_slashes=False)
def get_points():
    # used to infere the vertices, normals, triangles and colors

    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

    # load the params dictionary for the model
    idx = request.json["idx"]
    added_edges = request.json["added_edges"]
    added_nodes = request.json["added_nodes"]

    with open("input/data/threed_ssg/threed_ssg_subset_val.pkl", "rb") as f:
        data = pickle.load(f)

    if idx == "custom":
        scan_id = ["0cac75c6-8d6f-2d13-8d35-f0128b4fb7a9"]
        graph = dgl.DGLGraph()
    else:
        scan_id = [data[idx][1]]
        graph = data[idx][0]
        idx_nodes_not_shapenet = []
        shapenet_labels = get_shapenet_labels()
        node_labels = get_labels(
            graph=graph, dataset_name="threed_ssg", labels_type="nodes")
        for node in range(graph.number_of_nodes()):
            if node_labels[node] not in shapenet_labels:
                idx_nodes_not_shapenet.append(node)
        graph.remove_nodes(idx_nodes_not_shapenet)

    # add nodes
    for i in range(len(added_nodes)):
        id = int(added_nodes[i]["data"]["id"])
        global_id = int(added_nodes[i]["data"]["global_id"])
        graph = dgl.add_nodes(graph, 1, {'id': torch.tensor([id]),
                                         "global_id": torch.tensor([global_id])})

    # add edges
    for i in range(len(added_edges)):
        src = int(added_edges[i]["data"]["source"])
        src_idx = (graph.ndata["id"] == src).nonzero(
            as_tuple=True)[0]
        dst = torch.tensor([int(added_edges[i]["data"]["target"])])
        dst_idx = (graph.ndata["id"] == dst).nonzero(
            as_tuple=True)[0]
        label = torch.tensor([int(added_edges[i]["data"]["label_code"])])
        graph = dgl.add_edges(graph, src_idx, dst_idx, {'feat': label})

    # get_scene_graph_diagram(
    #     graph=graph, dataset_name="threed_ssg", filename="ciao.jpg", random_layout=True)

    # graph
    if graph.number_of_nodes() > 5:
        root_dir = "output/final/fullset/"
    else:
        root_dir = "output/final/subset/"

    # path to the saved file
    model_type = request.json["model"]
    print(model_type)
    if model_type == "gph":
        root_dir = root_dir + "GPH"
    elif model_type == "gnn":
        root_dir = root_dir + "GNN"
    elif model_type == "gnna":
        root_dir = root_dir + "GNNA"
    elif model_type == "gtn":
        root_dir = root_dir + "GTN"

    with open(os.path.join(root_dir, "config.json")) as f:
        params = json.load(f)

    params["use_prior"] = False
    params["dataset"] = 'threed_ssg_subset'  # for inference

    # select device
    device = select_device(params["device"])
    params['device'] = device

    # expand the graph attributes
    graph = laplacian_positional_encoding(
        graph, params["pe"]["pe_dim"]).to(device)

    scene, camera_target = scene_generation(graph=graph, scan_id=scan_id, device=device, params=params,
                                            root_dir=root_dir, path_to_shapenet=None)

    return {"pcd": scene, "camera_target": camera_target}


@app.route("/api/threedssg_vocabularies", methods=["GET"])
def get_vocabularies():
    rel_vocab = get_vocabulary(
        "input/data/threed_ssg/raw/vocab/relationships.tsv")

    obj_vocab = get_vocabulary(
        "input/data/threed_ssg/raw/vocab/objects.tsv", typeof="labels")

    obj_colors = get_vocabulary(
        "input/data/threed_ssg/raw/vocab/objects.tsv", typeof="colors")

    shapenet_vocab_labels, shapenet_vocab_colors = {}, {}
    shapenet_labels = get_shapenet_labels()

    for key in obj_colors.keys():
        if obj_vocab[key] in shapenet_labels:
            shapenet_vocab_labels[key] = obj_vocab[key]
            shapenet_vocab_colors[key] = "#"+str(rgb_to_hex(obj_colors[key]
                                                            [0], obj_colors[key][1], obj_colors[key][2]))

    return {"rel": rel_vocab, "obj": shapenet_vocab_labels,  "obj_colors": shapenet_vocab_colors}


@ app.route("/api/generate_3d_scene", methods=["POST"], strict_slashes=False)
def generate_3d_scene():

    try:

        # remove library warnings
        warnings.filterwarnings("ignore")
        o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

        # load the params dictionary for the model
        model_type = request.json["model"]

        if model_type == "GCN":
            root_dir = "output/threed_ssg/graphto3d/subset"
            with open(os.path.join(root_dir, "config.json")) as f:
                params = json.load(f)

        elif model_type == "GCN+A":
            root_dir = "output/threed_ssg/graphto3d_attention/subset"
            with open(os.path.join(root_dir, "config.json")) as f:
                params = json.load(f)

        elif model_type == "GTN":
            # root_dir = "output/threed_ssg/graph_transformer/subset"
            root_dir = "output/new_tests/graph_transformer"
            with open(os.path.join(root_dir, "config.json")) as f:
                params = json.load(f)

        params["use_prior"] = True

        # select device
        device = select_device(params["device"])
        params['device'] = device

        # load user inputs
        objects = request.json["objects"]
        relationships = request.json["relationships"]
        if objects == [] or relationships == []:
            return "Empty!"

        # type of graph
        scan_id = request.json["scan_id"]

        # build the graph
        graph = dgl.DGLGraph()
        for i in range(len(objects)):
            graph = dgl.add_nodes(graph, 1, {'id': torch.tensor(
                [objects[i][1]]), "global_id": torch.tensor([objects[i][0]])})

        edge_features = []
        for i in range(len(relationships)):
            graph.add_edges(relationships[i][0], relationships[i][1])
            edge_features.append(relationships[i][2])

        edge_features = torch.tensor(edge_features)
        graph.edata['feat'] = edge_features
        scan_id = ["f62fd5fd-9a3f-2f44-883a-1e5cf819608e"]

        # add positional encoding
        graph = laplacian_positional_encoding(
            graph, params["pe"]["pe_dim"]).to(device)

        scene_generation(graph=graph, scan_id=scan_id, device=device, params=params,
                         root_dir=root_dir, path_to_shapenet=None)

        return "Success!"

    except Exception as e:

        print(e)

        return "Error!"


# run with python3 server.py
if __name__ == "__main__":
    app.run(debug=True)
    # generate_3d_scene()
