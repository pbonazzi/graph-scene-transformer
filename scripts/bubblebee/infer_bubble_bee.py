# import libraries
import os
import time
import numpy as np
import argparse
import json
import warnings
import open3d as o3d
import torch
from dgl.dataloading.pytorch import GraphDataLoader
import pickle
from network.models.model import BubbleBee
# import custom methods
from scripts.inference import inference
from network.models.model import BubbleBee
from render.colors import bcolors
from scripts.preprocess_dgl import Dataset3DSSG
from config import OUTPUT_DIR, select_device
# import libraries
import numpy as np
import os
import random
import glob
import torch
from torch.utils.data import DataLoader

# import custom methods
from render.colors import bcolors
from render.utils import SceneInfoContainer, compute_box_3d
from render.two_d import get_scene_graph_diagram
from render.three_d import visualize, get_shapenet_labels, get_object_mesh, get_lineset, visualize_O3DVisualizer
from data.normalization import Normalizer
from network.models.model import BubbleBee
from network.metrics.utils import constraints_validation, sinkhorn_wassersein, normalized_sum, get_diversity
from network.metrics.iou import iou_boxes
from data.utils import get_labels



def total_param(params: dict) -> int:
    model = BubbleBee(params)
    counter = 0
    for param in model.parameters():
        counter += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', counter)
    return counter


def main(args):

    try:
        root : str = os.path.join(OUTPUT_DIR, "bubblebee", "results", args.name)
        exists : bool = os.path.isdir(root)
        os.makedirs(root, exist_ok=True)

        if exists and args.load:
            params = load_config(os.path.join(root, "config.yaml"))
        else:
            params = load_config(args.config)
            params["dataset"] = args.dataset
            params['root'] = root
            params['total_param'] = total_param(params)
            print(f"{bcolors.OKGREEN} Configuration file found. {bcolors.ENDC}")

    except TypeError:
        print(
            f"{bcolors.FAIL}[E] Please pass a valid config file.{bcolors.ENDC}")
        return
    except FileNotFoundError:
        print(
            f"{bcolors.FAIL}[E] There is no config file in this path.{bcolors.ENDC}")
        return

    # copy data object to params
    if args.load_run is None:
        data = config["data"]
        for key in data.keys():
            params[key] = data[key]
    else:
        data = config

    # select device
    device = select_device(config['device'])

    params['device'] = device
    params["tests"] = config['tests']
    params['inference'] = config['inference']

    if params["encoder"]['hidden_dimension'] % params["encoder"]['n_of_attention_heads'] != 0 or \
            params["decoder"]['hidden_dimension'] % params["decoder"]['n_of_attention_heads'] != 0:
        raise ValueError(f"{bcolors.FAIL}[E] The hidden_dim dimensions must be perfectly divisible by the number of "
                         f"heads. It's a convection for multi-head attention, sorry !{bcolors.ENDC}")

    # ignore warnings
    if not params["tests"]["warnings"]:
        print(
            f"{bcolors.WARNING}[W] You have disabled libraries' warnings.{bcolors.ENDC}")
        warnings.filterwarnings("ignore")
        o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

    # select dataset
    dataset_name = "threed_ssg_subset"
    params["dataset"] = "threed_ssg_subset"


    # import dataset
    with open(os.path.join(OUTPUT_DIR, "processed_data", "threed_ssg_subset_train.pkl"), 'rb') as f:
        train_set = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, "processed_data", "threed_ssg_subset_val.pkl"), 'rb') as f:
        val_set = pickle.load(f)

    train_loader = GraphDataLoader(
        train_set, batch_size=params["batch_size"], shuffle=True, drop_last=False, num_workers=4)

    val_loader = GraphDataLoader(
        val_set, batch_size=params["batch_size"], shuffle=True, drop_last=False, num_workers=4)

    load = False
    # select output directory
    if args.load_run is not None:
        root_dir = args.load_run
        load = True
    elif len(data["checkpoints_path"]) > 0:
        root_dir = data["checkpoints_path"]
        load = True
    elif args.output_path is not None:
        root_dir = args.output_path
    else:
        root_dir = config["output_dir"]
        root_dir = os.path.join(root_dir, time.strftime('%y_%m_%d_%Hh%Mm%Ss'))

    # create output directory
    os.makedirs(root_dir, exist_ok=True)
    params['root_dir'] = root_dir
    params["deterministic"] = config['tests']['deterministic']
    params['total_param'] = total_param(params)

def load_checkpoint(model, check_dir, device):
    # load the model
    files = glob.glob(check_dir + '/*.pkl')
    assert len(files) > 0, print(
        f"{bcolors.WARNING} No checkpoint was found under this directory path {check_dir}{bcolors.ENDC}")

    epoch_nb_list = []
    # loop over all saved models
    for i, file in enumerate(files):
        epoch_nb = file.split('_')[-1]
        epoch_nb = int(epoch_nb.split('.')[0])  # extract the number
        epoch_nb_list.append(epoch_nb)

    print(
        f"{bcolors.OKGREEN}Which epoch from the list below you'd like to load the checkpoint?{bcolors.ENDC}")
    print(epoch_nb_list)
    epoch_nb = int(input())
    # load from selected checkpoint
    idx = epoch_nb_list.index(epoch_nb)
    if idx is not None:
        file = files[idx]
    else:
        raise NameError('Checkpoint Not Found')
    print(f"Loading trained model of epoch {epoch_nb}...")
    model.load_state_dict(torch.load(
        file, map_location=torch.device(device)))

    return model, epoch_nb


def load_statistics(train_stats_out, epoch_nb, model, train_loader):
    load = False
    filename = os.path.join(
        train_stats_out, "train_stats.npz")
    if os.path.exists(filename) and os.stat(filename).st_size > 0:
        npzfile = np.load(filename)
        epoch_num = npzfile['epoch_nb']
        # check if the lastest checkpoint stats loaded
        if epoch_nb == epoch_num:
            print(
                f"{bcolors.OKGREEN} Loading train stats from saved posterior distribution for epoch {epoch_num}...{bcolors.ENDC}")
            mean_node_est = npzfile['mean_node_est']
            cov_node_est = npzfile['cov_node_est']
            mean_edge_est = npzfile['mean_edge_est']
            cov_edge_est = npzfile['cov_edge_est']
            posterior_statistics = [
                mean_node_est, cov_node_est, mean_edge_est, cov_edge_est]
            load = True
    if not load:
        print(
            f"{bcolors.OKGREEN} Computing train stats for model of epoch {epoch_nb}...{bcolors.ENDC}")
        mean_node_est, cov_node_est, mean_edge_est, cov_edge_est = model.collect_train_statistics(train_loader,
                                                                                                  plot=True)
        posterior_statistics = [mean_node_est,
                                cov_node_est, mean_edge_est, cov_edge_est]

        np.savez(filename, epoch_nb=epoch_nb, mean_node_est=mean_node_est,
                 cov_node_est=cov_node_est, mean_edge_est=mean_edge_est, cov_edge_est=cov_edge_est)
    assert posterior_statistics is not None
    return posterior_statistics


def inference(dataset: DGL, params: dict, root_dir: str, path_to_shapenet: str):

    # user defined
    visualize_mode = False
    stats_mode = True  # compute only the statistics
    avg_iter = 10

    #  directories
    log_dir = os.path.join(root_dir, "logs")
    check_dir = os.path.join(root_dir, "checkpoints")
    fig_dir = os.path.join(root_dir, "figures_shapenet")
    results_txt = os.path.join(root_dir, "inference")
    train_stats_out = os.path.join(root_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(train_stats_out, exist_ok=True)

    # deterministic / non deterministic
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    device = params['device']
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    # initialize the model
    model = BubbleBee(params)
    model = model.to(device)
    model.eval()

    # load scene
    train_set, val_set = dataset.train, dataset.val

    # load iterable input
    train_loader = DataLoader(
        train_set, batch_size=params["batch_size"], shuffle=True, collate_fn=dataset.__getitem__)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, collate_fn=dataset.__getitem__)

    # load shapenet labels
    shapenet_labels = get_shapenet_labels(path_to_shapenet)
    selected_obj_to_render = ['ball', 'basket', 'bench', 'bed', 'cabinet', 'chair', 'armchair',
                              'desk', 'door', 'floor', 'picture', 'sofa', 'couch', 'commode', 'monitor',
                              'stool', 'tv', 'table', 'pillow', 'plant', 'carpet']

    # load checkpoint
    model, epoch_nb = load_checkpoint(model, check_dir, device)

    # load/compute statistics
    if not params["use_prior"]:
        posterior_statistics = load_statistics(
            train_stats_out, epoch_nb, model, train_loader)
    else:
        posterior_statistics = None

    # metrics
    gbl_sw_distance, gbl_iou_boxes_pred, gbl_iou_boxes_gt = 0, 0, 0
    glb_diversity_size, glb_diversity_location, glb_diversity_angle = 0, 0, 0

    if stats_mode:
        epoch_accuracy = {
            "left": [],
            "right": [],
            "front": [],
            "behind": [],
            "higher": [],
            "lower": [],
            "bigger": [],
            "smaller": []
        }
    with torch.no_grad():
        for graph, ids in val_loader:
            print(
                f"{bcolors.WARNING} Loading graph {ids[0]}{bcolors.ENDC}")
            epoch_diversity = []
            for j in range(int(avg_iter)):

                # inference
                _, scores, _ = model.inference(
                    graphs=graph, ids=ids, posterior_statistics=posterior_statistics, var_scale=1)

                # denormalize predictions
                cpu_scores = scores.cpu().detach().clone().numpy()
                Normal = Normalizer(params=params)
                GraphScene = SceneInfoContainer(
                    graph=graph, scores=cpu_scores, params=params, N=Normal)

                if stats_mode:
                    # edge feature constraints
                    epoch_accuracy = constraints_validation(
                        accuracy=epoch_accuracy, g=graph, scores=cpu_scores, params=params)

                    # iou boxes
                    pred, gt = iou_boxes(
                        graph=graph, scores=cpu_scores, params=params)
                    gbl_iou_boxes_pred += pred
                    gbl_iou_boxes_gt += gt

                    # wassersein distance
                    gbl_sw_distance += sinkhorn_wassersein(
                        graph=graph, scores=cpu_scores, params=params).detach().item()

                    # diversity
                    epoch_diversity.append(cpu_scores)

                # visualization
                if visualize_mode:
                    score_lines_list, score_mesh_list = [], []
                    idx_fp_nodes = []
                    nodes_colors = get_labels(
                        graph=graph, dataset_name=params["dataset"], labels_type="nodes_colors")
                    for node in range(GraphScene.nodes):

                        # name of the objects
                        obj_name = GraphScene.labels[node]
                        color = nodes_colors[node]

                        # draw the boxes
                        scores_points = compute_box_3d(dim=GraphScene.dim_scores[node], location=GraphScene.loc_scores[node],
                                                       angle=np.argmax(GraphScene.ori_scores[node]) * 90 / GraphScene.oridim)
                        lines = get_lineset(points=scores_points, rgb=[
                                            0, 0, 1])  # blue is score
                        score_lines_list.append(lines)

                        # draw the meshes
                        if obj_name in selected_obj_to_render and obj_name in shapenet_labels:
                            print(
                                f'{obj_name}: groundtruth dimension: {GraphScene.dim_targets[node]}; prediction dimension: {GraphScene.dim_scores[node]}')
                            mesh = get_object_mesh(label=obj_name,
                                                   color=color,
                                                   path_to_shapenet=path_to_shapenet,
                                                   dimension=GraphScene.dim_scores[node],
                                                   location=GraphScene.loc_scores[node],
                                                   angle=np.argmax(GraphScene.ori_scores[
                                                       node]).item() * 90 / GraphScene.oridim,
                                                   direction=GraphScene.directions[node])

                            if mesh is not None:
                                score_mesh_list.append(mesh)
                        else:
                            idx_fp_nodes.append(node)

                    # move scene graph diagram to the folder
                    if j == 1:
                        graph_for_picture = graph.clone()
                        graph_for_picture.remove_nodes(idx_fp_nodes)
                        if graph_for_picture.number_of_nodes() > 3:
                            get_scene_graph_diagram(
                                graph=graph, dataset_name="threed_ssg", filename=os.path.join(fig_dir, ids[0] + "scene_diagram.png"))
                    if len(score_mesh_list) > 3:
                        visualize(score_mesh_list, scan_id=ids[0],
                                  filename=os.path.join(fig_dir, ids[0]+"RUN_"+str(j)), save_img=True)
                    else:
                        break

            if stats_mode:
                size, location, angle = get_diversity(
                    epoch_diversity, params)

                glb_diversity_size += size
                glb_diversity_location += location
                glb_diversity_angle += angle
    if stats_mode:
        gbl_accuracy = {
            "left_right": normalized_sum(epoch_accuracy["left"], epoch_accuracy["right"]),
            "front_behind": normalized_sum(epoch_accuracy["front"], epoch_accuracy["behind"]),
            "higher_lower": normalized_sum(epoch_accuracy["higher"], epoch_accuracy["lower"]),
            "bigger_smaller": normalized_sum(epoch_accuracy["bigger"], epoch_accuracy["smaller"])}

        glb_diversity_size /= len(val_loader)
        glb_diversity_location /= len(val_loader)
        glb_diversity_angle /= len(val_loader)
        gbl_sw_distance /= (len(val_loader)*avg_iter)
        gbl_iou_boxes_gt /= (len(val_loader)*avg_iter)
        gbl_iou_boxes_pred /= (len(val_loader)*avg_iter)

        gbl_accuracy_mean = (gbl_accuracy["left_right"] + gbl_accuracy["front_behind"] +
                             gbl_accuracy["higher_lower"] + gbl_accuracy["bigger_smaller"]) / 4

        # write the results of inference
        with open(results_txt + '.txt', 'a+') as f:
            f.write(
                """
                EPOCH {:d} METRICS\n

                iou gt and pred \n
                {:.2f} & {:.2f}\n

                sw \n
                {:.2f}\n
                
                diversity \n
                size , location, angle \n
                {:.2f} & {:.2f} & {:.2f} \n
                
                accuracy  \n
                mean, higher/lower, bigger/smaller, left/right, front/behind \n
                {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \n


                """.format(epoch_nb,

                           gbl_iou_boxes_gt,
                           gbl_iou_boxes_pred,

                           gbl_sw_distance,

                           glb_diversity_size,
                           glb_diversity_location,
                           glb_diversity_angle,

                           gbl_accuracy_mean,
                           gbl_accuracy["higher_lower"],
                           gbl_accuracy["bigger_smaller"],
                           gbl_accuracy["left_right"],
                           gbl_accuracy["front_behind"]

                           ))

    return






if __name__ == "__main__":
    import inspect
    import sys
    import torch
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from scripts.preprocess_dgl import Dataset3DSSG

    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1

    torch.set_num_threads(10)  # intraop parallelism
    torch.set_num_interop_threads(10)  # interop parallelism

    # parser
    parser = argparse.ArgumentParser(
        description='I solemnly swear that I am up to no good.')

    parser.add_argument('--name', '--c',
                        help="Path to configuration file, see config/config.json")
    parser.add_argument('--output', '--o',
                        help='Path to output directory')
    args = parser.parse_args()


    main(args)
