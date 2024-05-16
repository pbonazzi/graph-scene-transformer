# import libraries
import numpy as np
import torch
from tqdm import tqdm

# import custom methods
from network.metrics.loss import calculate_loss
from network.metrics.utils import constraints_validation, sinkhorn_wassersein, normalized_sum
from network.metrics.iou import iou_boxes

def evaluate_network(model: torch.nn.Module, data_loader: torch.utils.data.dataloader.DataLoader, params: dict) -> tuple:
    """ Evaluation mode
    """

    # tell the model that we are in test mode
    model.eval()

    # initialize variables
    epoch_test_loss, epoch_loss_cat, epoch_loss_ori, epoch_loss_dim, epoch_loss_loc, epoch_loss_kl = 0, 0, 0, 0, 0, 0

    # disabled gradient calculation to reduce memory consumption
    with torch.no_grad():
        for i, (batch_graphs, ids) in tqdm(enumerate(data_loader), position=2, disable=True):

            # evaluate
            if params["dataset"] == "3dssg":
                ids = [str(id).split("_")[0] for id in ids]
            
            latent_graph, scores, targets, mu_node, logvar_node, mu_edge, logvar_edge = model(
                batch_graphs, ids)

            # calculate and save losses
            local_loss, loss_deconstructed, loss_weights = calculate_loss(
                model, params, scores, targets)

            # cumulate losses for tensorboard
            if not params['decoder']["output_object_code"]:
                loss_ori, loss_dim, loss_loc, KL_loss = loss_deconstructed[
                    'ori'], loss_deconstructed['dim'], loss_deconstructed['loc'], loss_deconstructed['kl']
            else:
                loss_ori, loss_dim, loss_loc, KL_loss, loss_cat = loss_deconstructed[
                    'ori'], loss_deconstructed['dim'], loss_deconstructed['loc'], loss_deconstructed['kl'], loss_deconstructed['cat']
                epoch_loss_cat += loss_cat

            epoch_loss_ori += loss_ori
            epoch_loss_dim += loss_dim
            epoch_loss_loc += loss_loc
            epoch_loss_kl += KL_loss
            epoch_test_loss += local_loss.detach().item()
            if params["overfit"]:
                break
    # divide metrics by number of iter+1 and assign the result to them.
    epoch_test_loss /= (i + 1)
    epoch_loss_cat /= (i + 1)
    epoch_loss_ori /= (i + 1)
    epoch_loss_dim /= (i + 1)
    epoch_loss_loc /= (i + 1)
    epoch_loss_kl /= (i + 1)

    # epoch_loss_deconstructed = [epoch_loss_ori, epoch_loss_dim, epoch_loss_loc, epoch_loss_kl, epoch_loss_cat]
    epoch_loss_deconstructed = {
        'ori': epoch_loss_ori, 'dim': epoch_loss_dim, 'loc': epoch_loss_loc, 'kl': epoch_loss_kl, 'cat': epoch_loss_cat}
    return epoch_test_loss, epoch_loss_deconstructed, loss_weights


def compute_eval_metrics(model: torch.nn.Module, data_loader: torch.utils.data.dataloader.DataLoader, params: dict, Normalizer) -> list:
    """ Evaluation mode
    """

    # tell the model that we are in test mode
    model.eval()

    # initialize variables
    epoch_iou_boxes, epoch_sw_distance = 0, 0

    # accuracy dictionary
    accuracy = {
        "left": [],
        "right": [],
        "front": [],
        "behind": [],
        "higher": [],
        "lower": [],
        "bigger": [],
        "smaller": []
    }

    # disabled gradient calculation to reduce memory consumption
    with torch.no_grad():
        for i, (batch_graphs, ids) in tqdm(enumerate(data_loader), position=2, disable=True):

            # evaluate
            latent_graph, scores, targets, mu_node, logvar_node, mu_edge, logvar_edge = model(
                batch_graphs, ids)
            # de normalize features)
            cpu_scores = scores.cpu().detach().clone().numpy()
            Normalizer.denormalize(data=cpu_scores[:, :3], feature="dimension")
            Normalizer.denormalize(data=cpu_scores[:, 3:6], feature="location")
            # edge feature constraints
            accuracy = constraints_validation(
                accuracy=accuracy, g=batch_graphs, scores=cpu_scores, params=params)
            # iou boxes
            epoch_iou_boxes += iou_boxes(graph=batch_graphs,
                                         scores=cpu_scores, params=params, Normalizer=Normalizer)[0]
            # wassersein distance
            epoch_sw_distance += sinkhorn_wassersein(
                graph=batch_graphs, scores=cpu_scores, params=params).detach().item()

        epoch_accuracy = {
            "left_right": normalized_sum(accuracy["left"], accuracy["right"]),
            "front_behind": normalized_sum(accuracy["front"], accuracy["behind"]),
            "higher_lower": normalized_sum(accuracy["higher"], accuracy["lower"]),
            "bigger_smaller": normalized_sum(accuracy["bigger"], accuracy["smaller"])}
        epoch_sw_distance /= (i + 1)
        epoch_iou_boxes /= (i + 1)
        epoch_metrics = [epoch_accuracy, epoch_sw_distance, epoch_iou_boxes]

        return epoch_metrics
