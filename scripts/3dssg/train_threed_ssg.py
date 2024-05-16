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

# PARAMS
params = {
    "overfit": False,
    "bbox_embed": 48,
    "hidden_dim": 64,
    "number_of_edge_classes": 27,
    "lr_init": 5e-3,
    "lr_min": 0.0,
    "lr_reduce_factor": 0.85,
    "lr_schedule_patience": 3,
    "weight_decay": 0.0,
    "betas": [0.9, 0.98],
    "batch_size": 64,
    "epochs": 100,
    "cross_entropy_weight": False
}

def main():
    
    parser = argparse.ArgumentParser(
    description='I solemnly swear that I am up to no good.')

    parser.add_argument('--name', '--n', required=True,
                        help="Name of the run")
    args = parser.parse_args()

    root = os.path.join(OUTPUT_DIR, "results",args.name)
    os.makedirs(root, exist_ok=True)
    check_dir = os.path.join(root, "checkpoints")
    os.makedirs(check_dir, exist_ok=True)
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # config
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(params, f, sort_keys=True, indent=4)

    # logs
    writer = SummaryWriter(log_dir=log_dir)
    device = select_device()
    model = Model3DSSG(params=params, device=device)
    model = model.to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr_init'],
        betas=params['betas'],
        weight_decay=params['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=params['lr_reduce_factor'],
                                                           patience=params[
                                                               'lr_schedule_patience'],
                                                           verbose=True)

    with open(os.path.join(OUTPUT_DIR, "3dssg", "processed_data", "3dssg_train.pkl"), 'rb') as f:
        train_set = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, "3dssg", "processed_data", "3dssg_val.pkl"), 'rb') as f:
        val_set = pickle.load(f)

    train_loader = GraphDataLoader(
        train_set, batch_size=params["batch_size"], shuffle=True, drop_last=False, num_workers=4)
    
    # weights
    weights = None
    if params["cross_entropy_weight"]:
        largest_class = max(train_set.classes.values())
        weights = []
        for key in train_set.classes:
            if train_set.classes[key]==0: 
                weights.append(largest_class)
                continue
            weights.append(largest_class/train_set.classes[key])
        weights = torch.tensor(weights).to(device)
    

    val_loader = GraphDataLoader(
        val_set, batch_size=params["batch_size"], shuffle=True, drop_last=False, num_workers=4)

    with tqdm(range(params['epochs']), position=0, leave=False) as t:
        for epoch in t:
            start = time.time()

            # anomaly detection
            torch.autograd.set_detect_anomaly(True)

            # model train
            model = model.to(device)
            model.train()
            epoch_train_loss = 0

            for ite, (batch_graphs, ids) in tqdm(enumerate(train_loader), disable=False):

                optimizer.zero_grad()
                scores, targets = model(batch_graphs)
                train_loss = model.loss(scores, targets,"CrossEntr", weights)
                train_loss.backward()
                epoch_train_loss += train_loss.item()

                optimizer.step()

                if params["overfit"]:
                    break

            # model val
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for ite, (batch_graphs, ids) in tqdm(enumerate(val_loader), disable=True):
                    optimizer.zero_grad()

                    scores, targets = model(batch_graphs)

                    val_loss = model.loss(scores, targets,
                                          "CrossEntr", weights).to(device)

                    epoch_val_loss += val_loss.item()

                    if params["overfit"]:
                        break

            # save checkpoint
            torch.save(model.state_dict(), '{}.pkl'.format(
                check_dir + "/epoch_" + str(epoch)))
            previous_epoch = os.path.join(
                check_dir, "epoch_" + str(epoch-1)+".pkl")
            if os.path.isfile(previous_epoch):
                os.remove(previous_epoch)

            # writer
            t.set_description('Epoch %d' % epoch)
            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss)
            writer.add_scalar('epoch_val_loss', epoch_val_loss, epoch)
            writer.add_scalar('epoch_train_loss', epoch_train_loss, epoch)

            # scheduler
            scheduler.step(metrics=epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params["lr_min"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break


if __name__ == "__main__":
    import inspect
    import sys
    import torch
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)
    from render.colors import bcolors
    from scripts.preprocess_dgl import Dataset3DSSG
    from network.models.model_3dssg import Model3DSSG
    from config.paths import  OUTPUT_DIR, THREED_SSG_PLUS
    from scripts import select_device

    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1

    torch.set_num_threads(10)  # intraop parallelism
    torch.set_num_interop_threads(10)  # interop parallelism

    main()
