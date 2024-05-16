# import libraries
import torch
from tqdm import tqdm
import os
import dgl
import pdb

# import custom methods
from network.metrics.loss import calculate_loss
from render.utils import SceneInfoContainer
import render.logging as tb
from data.normalization import Normalizer
from render.colors import bcolors


def train_epoch(model: torch.nn.Module, optimizer: torch.optim,
                data_loader, epoch: int, params: dict, writer: object):
    """ Training model """

    # initialize training mode
    model.train()

    # initialize variables
    epoch_loss = 0

    for ite, (batch_graphs, ids) in tqdm(enumerate(data_loader), disable=True):

        # gradient
        optimizer.zero_grad()

        # model - forward
        if params["dataset"] == "3dssg": # subset , for full skip
            ids = [str(id).split("_")[0] for id in ids] 

        enhanced_graphs, scores, targets, mu_node, logvar_node, mu_edge, logvar_edge = model(
            batch_graphs, ids)

        # model - loss
        local_loss, loss_deconstructed, loss_weights = calculate_loss(
            model=model, params=params, scores=scores, targets=targets)
        epoch_loss += local_loss.detach().item()

        # tensorboard - open3d boxes
        stepsize = 10
        # for i in range(len(ids)):
        #     if ids[i] == view and epoch % stepsize == 0:
        #         graph = dgl.unbatch(enhanced_graphs)[i].to("cpu")
        #         cpu_scores = scores.cpu().detach().numpy()
        #         Normal = Normalizer(params=params)
        #         GraphScene = SceneInfoContainer(
        #             graph=graph, scores=cpu_scores, params=params, N=Normal)
        #         print(
        #             f"{bcolors.OKGREEN} Adding scene to tensorboard...{bcolors.ENDC}")
        #         tb.add_nodes_open3d(GraphScene=GraphScene,
        #                             representation=params["render"]["tensorboard_representation"],
        #                             writer=writer,
        #                             step=int(epoch / stepsize),
        #                             iter=epoch)
        #         print(
        #             f"{bcolors.OKGREEN} Adding scene to tensorboard Done...{bcolors.ENDC}")
        # tensorboard - subplots
        # if tb.is_relevant(ratio=(epoch / params["epochs"])):
        #     writer.add_scalar(os.path.join(
        #         "train", "orientation"), loss_deconstructed['ori'], epoch)
        #     writer.add_scalar(os.path.join("train", "dimension"),
        #                       loss_deconstructed['dim'], epoch)
        #     writer.add_scalar(os.path.join("train", "location"),
        #                       loss_deconstructed['loc'], epoch)
        #     writer.add_scalar(os.path.join("train", "KL_loss"),
        #                       loss_deconstructed['kl'], epoch)
        #     if params["decoder"]["output_object_code"]:
        #         writer.add_scalar(os.path.join(
        #             "train", "category"), loss_deconstructed['cat'], epoch)

        #     writer.add_scalar(os.path.join("train", "orientation loss sigma"),
        #                       loss_weights[0], epoch)
        #     writer.add_scalar(os.path.join("train", "dimension loss sigma"),
        #                       loss_weights[1], epoch)
        #     writer.add_scalar(os.path.join("train", "location loss sigma"),
        #                       loss_weights[2], epoch)
        #     writer.add_scalar(os.path.join("train", "KL loss weight"),
        #                       loss_weights[3], epoch)
        #     if params["decoder"]["output_object_code"]:
        #         writer.add_scalar(os.path.join(
        #             "train", "category loss sigma"), loss_weights[4], epoch)

        # model - backward
        local_loss.backward()

        if params['gradient_clipping']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params['clipping_value'])
        optimizer.step()
        
        if params["n_of_graphs"] > ite:
            break

    epoch_loss /= (ite + 1)
    return epoch_loss, optimizer
