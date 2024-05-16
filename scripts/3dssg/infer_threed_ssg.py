import os
import time
import csv
import torch
from dgl.dataloading.pytorch import GraphDataLoader
import dgl
import pickle
from tqdm import tqdm
import json, argparse

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(
    description='I solemnly swear that I am up to no good.')

    parser.add_argument('--path', '--p', required=True,
                        help="Path to folder")
    parser.add_argument('--chk', '--c', required=True,
                        help="Path to folder")
    args = parser.parse_args()
    
    
    img_dir = os.path.join(args.path, "images")
    os.makedirs(img_dir, exist_ok=True)
    log_dir = os.path.join(args.path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # logs
    writer = SummaryWriter(log_dir=log_dir)
    device = select_device()
    with open(args.path + "config.json") as f:
        config = json.load(f)
        params = config

    model = Model3DSSG(params=params, device=device)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.path, "checkpoints", "epoch_"+args.chk+".pkl"), map_location=torch.device(device)))
    print(f"{bcolors.OKGREEN} Loaded checkpoint{bcolors.ENDC}")
        
        
    # dataset
    with open(os.path.join(OUTPUT_DIR, "processed_data", "threed_ssg_subset_val.pkl"), 'rb') as f:
        val_set = pickle.load(f)
    val_loader = GraphDataLoader(
        val_set, batch_size=1, shuffle=True, drop_last=False, num_workers=4)


    # model val
    model.eval()
    with torch.no_grad():
        for ite, (batch_graphs, ids) in tqdm(enumerate(val_loader), disable=True):
            
            
            graph_without_edges = batch_graphs.clone()
            graph_without_edges.remove_edges(graph_without_edges.edges("all")[2])

            inferred_graphs = model.inference(graph_without_edges)
            get_scene_graph_diagram(inferred_graphs, filename=os.path.join(img_dir, ids[0]+'infer'))

            if params["overfit"]:
                break


if __name__ == "__main__":
    import inspect
    import sys
    import torch
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)
    
    from scripts.preprocess_dgl import Dataset3DSSG
    from network.models.model_3dssg import Model3DSSG
    from scripts import  select_device
    from render.colors import bcolors
    from render.two_d import get_scene_graph_diagram
    from config.paths import  OUTPUT_DIR, THREED_SSG_PLUS

    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1

    torch.set_num_threads(10)  # intraop parallelism
    torch.set_num_interop_threads(10)  # interop parallelism

    main()
