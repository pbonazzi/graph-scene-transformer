# import libraries
import numpy as np
import os
import random
import glob
import torch
import dgl
import open3d as o3d

# import custom methods
from render.colors import bcolors
from render.utils import SceneInfoContainer
from render.three_d import get_shapenet_labels, get_object_mesh
from data.normalization import Normalizer
from data.utils import get_labels
from network.models.model import BubbleBee
from render.colors import bcolors


def scene_generation(graph: dgl.graph, scan_id: str, device: torch.device, params: dict, root_dir: str, path_to_shapenet: str):
    #  directories
    log_dir = os.path.join(root_dir, "logs")
    check_dir = os.path.join(root_dir, "checkpoints")
    fig_dir = os.path.join(root_dir, "generated")
    train_stats_out = os.path.join(root_dir, "train_stats.npz")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # deterministic / non deterministic
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    # load the model
    model = BubbleBee(params)
    model = model.to(device)
    model.eval()
    model.dataset = params["dataset"]

    files = glob.glob(check_dir + '/*.pkl')
    if len(files) != 0:
        epoch_nb_list = []
        for file in files:
            epoch_nb = file.split('_')[-1]
            epoch_nb = int(epoch_nb.split('.')[0])  # extract the number
            epoch_nb_list.append(epoch_nb)

        epoch_nb = max(epoch_nb_list)
        latest_epoch = epoch_nb_list.index(epoch_nb)
        model.load_state_dict(torch.load(
            files[latest_epoch], map_location=torch.device(device)))
        model.device = device

        print(f"{bcolors.OKGREEN} Loaded checkpoint{bcolors.ENDC}",
              ":", max(epoch_nb_list))
    else:
        print(
            f"{bcolors.ERROR} No checkpoint was found under this directory path {bcolors.ENDC}")

    # load posterior
    posterior_statistics = None
    if not params["use_prior"]:
        npzfile = np.load(train_stats_out)
        print(
            f"{bcolors.OKGREEN} Loading train stats from saved posterior distribution...{bcolors.ENDC}")
        mean_node_est = npzfile['mean_node_est']
        cov_node_est = npzfile['cov_node_est']
        mean_edge_est = npzfile['mean_edge_est']
        cov_edge_est = npzfile['cov_edge_est']
        posterior_statistics = [
            mean_node_est, cov_node_est, mean_edge_est, cov_edge_est]

    shapenet_labels = get_shapenet_labels(path_to_shapenet)

    with torch.no_grad():

        # inference
        _, scores, _ = model.inference(
            graph, scan_id, posterior_statistics, user_mode=True, var_scale=1)

        # load attributes
        cpu_scores = scores.cpu().detach().numpy()
        Normal = Normalizer(params=params)
        GraphScene = SceneInfoContainer(
            graph=graph, scores=cpu_scores, params=params, N=Normal, targets=False)
        labels = get_labels(
            graph=graph, dataset_name=params["dataset"], labels_type="nodes")
        nodes_colors = get_labels(
            graph=graph, dataset_name=params["dataset"], labels_type="nodes_colors")

        # mesh = o3d.geometry.TriangleMesh()
        camera_target = np.sum(GraphScene.loc_scores,
                               axis=0)/graph.number_of_nodes()
        scene = []

        for node in range(graph.number_of_nodes()):

            # name of the objects
            obj_name = labels[node]

            # draw the meshes
            # if obj_name in selected_obj_to_render and obj_name in shapenet_labels:
            if obj_name in shapenet_labels:
                color = nodes_colors[node]

                obj = get_object_mesh(label=obj_name,
                                      color=color,
                                      path_to_shapenet=path_to_shapenet,
                                      dimension=GraphScene.dim_scores[node],
                                      location=GraphScene.loc_scores[node],
                                      angle=np.argmax(GraphScene.ori_scores[
                                          node]).item() * 90 / GraphScene.oridim,
                                      direction=1)

                if obj is not None:
                    #mesh += obj
                    color = [int(color[0]*255), int(color[1]*255),
                             int(color[2]*255)]
                    triangles_list = np.asarray(obj.triangles).tolist()
                    vertices_list = np.asarray(obj.vertices).tolist()
                    normals_list = np.asarray(obj.vertex_normals).tolist()
                    colors_list = np.asarray(obj.vertex_colors).tolist()
                    scene.append(
                        [triangles_list, vertices_list, color, normals_list, colors_list])

        return scene, camera_target.tolist()
