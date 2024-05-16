# import libraries
import numpy as np
import torch
import json
import os


def calculate_tri_mse_loss(model: torch.nn.Module, scores: torch.tensor, targets: torch.tensor):
    """ Aggregate the calculation of the loss for tri dimensional features passed as one 3xN matrix

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    scores_1 = scores[:, 0]
    targets_1 = targets[:, 0]

    scores_2 = scores[:, 1]
    targets_2 = targets[:, 1]

    scores_3 = scores[:, 2]
    targets_3 = targets[:, 2]

    loss_1 = model.loss(scores_1, targets_1, "MSE")
    loss_2 = model.loss(scores_2, targets_2, "MSE")
    loss_3 = model.loss(scores_3, targets_3, "MSE")
    loss = loss_1 + loss_2 + loss_3

    return loss


def calculate_loss(model: torch.nn.Module, params: dict, scores: torch.tensor, targets: torch.tensor, adj_weight_KL: bool = False):
    """ Calculate the loss for any model

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    total_loss = 0.0

    # dimension
    scores_dim = scores[:, :3]
    targets_dim = targets[:, :3]
    loss_dim = calculate_tri_mse_loss(
        model=model, scores=scores_dim, targets=targets_dim)
    epoch_loss_dim = loss_dim.detach().item()

    # location
    scores_loc = scores[:, 3:6]
    targets_loc = targets[:, 3:6]
    loss_loc = calculate_tri_mse_loss(
        model=model, scores=scores_loc, targets=targets_loc)
    epoch_loss_loc = loss_loc.detach().item()

    # orientation
    if params['orientation_loss']:
        scores_ori = scores[:, 6:params["n_of_orientation_bins"] + 6]
        targets_ori = targets[:, 6:params["n_of_orientation_bins"] + 6]
        loss_ori = model.loss(scores_ori, targets_ori,
                              "CrossEntr").to(params["device"])
        epoch_loss_ori = loss_ori.detach().item()
    else:
        loss_ori, epoch_loss_ori = 0.0, 0.0

    # KL loss
    if params['inference'] or params['deterministic']:
        KL_weight, loss_gauss = 0.0, 0.0
    else:
        mu, log_var = model.mu_node, model.logvar_node
        loss_gauss = -0.5 * \
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
        KL_weight = params["weight_KLD_loss"]

    loss_gauss_weighted = loss_gauss * KL_weight

    if not params['decoder']['output_object_code']:
        if params["multi_layer_loss"]:
            reconstruction_loss, sigma = model.multi_layer_loss(
                loss_list=[loss_ori, loss_dim, loss_loc])
            loss_weights = np.append(sigma.cpu().detach().numpy(), [KL_weight])
        else:
            reconstruction_loss = loss_ori + loss_dim + loss_loc
            loss_weights = [1, 1, 1, KL_weight]

        # loss_deconstructed = [epoch_loss_ori, epoch_loss_dim, epoch_loss_loc, loss_gauss]
        loss_deconstructed = {
            'ori': epoch_loss_ori, 'dim': epoch_loss_dim, 'loc': epoch_loss_loc, 'kl': loss_gauss}
        total_loss = reconstruction_loss + loss_gauss_weighted
        if loss_gauss > 20 or epoch_loss_ori > 5 or epoch_loss_dim > 5 or epoch_loss_loc > 5:
            log_dir = os.path.join(params['root_dir'], "logs")
            with open(log_dir + 'loss_log.txt', 'a+') as f:
                f.write("""Unusual Loss: \n
                    total loss={:.4f}\n
                    deconstructed loss: {}\n""".format(total_loss, loss_deconstructed))

        return total_loss, loss_deconstructed, loss_weights

    else:
        # category
        scores_cat = scores[:, params["n_of_orientation_bins"] + 6:]
        targets_cat = targets[:, params["n_of_orientation_bins"] + 6:]
        loss_cat = model.loss(scores_cat, targets_cat, "BCEwithLL")
        epoch_loss_cat = loss_cat.detach().item()

        if params["multi_layer_loss"]:
            reconstruction_loss, sigma = model.multi_layer_loss(
                [loss_cat, loss_ori, loss_dim, loss_loc])
            loss_weights = np.append(sigma.cpu().detach().numpy(), [KL_weight])
        else:
            reconstruction_loss = loss_cat + loss_ori + loss_dim + loss_loc
            loss_weights = [1, 1, 1, 1, KL_weight]

            # loss_deconstructed = [epoch_loss_ori, epoch_loss_dim, epoch_loss_loc, loss_gauss, epoch_loss_cat]
            loss_deconstructed = {
                'ori': epoch_loss_ori, 'dim': epoch_loss_dim, 'loc': epoch_loss_loc, 'kl': loss_gauss, 'cat': epoch_loss_cat}

        total_loss = reconstruction_loss + loss_gauss_weighted
        return total_loss, loss_deconstructed, loss_weights
