# import libraries
import torch
import torch.nn as nn


class MultiLossLayer(nn.Module):
    """ Unofficial implementation of
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    def __init__(self, var_types: list):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(len(var_types)))
        self.var_types = var_types

    def get_loss(self, loss_list):
        loss = 0
        for i in range(len(loss_list)):
            if self.var_types[i] == "con":
                loss = loss + \
                    loss_list[i] / (2 * self.sigma[i] ** 2) + \
                    torch.log(self.sigma[i])
            elif self.var_types[i] == "cat":
                loss = loss + \
                    loss_list[i] / (self.sigma[i] ** 2) + \
                    torch.log(self.sigma[i])
            else:
                raise ValueError
        return loss
