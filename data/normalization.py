# import libraries
import json
import torch


def denormalization(data: torch.tensor, mean: float, std: float):
    """
    Parameters
    (float) mean, std : range used for normalization
    (bool) is_normalized : if the data is normalized then it will be denormalized and vice versa.
    """

    for i in range(len(data)):
        data[i] = data[i] * std + mean

    return data


def normalization(data: torch.tensor, mean: float, std: float):
    """
    Parameters
    (float) mean, std : range used for normalization
    (bool) is_normalized : if the data is normalized then it will be denormalized and vice versa.
    """

    for i in range(len(data)):
        data[i] = (data[i] - mean) / std

    return data


class Normalizer:
    """ Util class to normalize and de normalize location and dimension

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    def __init__(self, dict_stats: dict):
        self.means_loc = dict_stats["train_location_mean"]
        self.std_loc = dict_stats["train_location_std"]
        self.means_dim = dict_stats["train_dimension_mean"]
        self.std_dim = dict_stats["train_dimension_std"]

    def denormalize(self, data: torch.tensor, feature: str):
        if feature == "location":
            for i in range(data.shape[1]):
                # print('loc mean', self.means_loc)
                # print('loc std', self.std_loc)
                data[:, i:i+1] = denormalization(data[:, i:i+1],
                                                 mean=self.means_loc[i], std=self.std_loc[i])
            return data
        if feature == "dimension":
            # print('dim mean', self.means_dim)
            # print('dim std', self.std_dim)
            for i in range(data.shape[1]):
                data[:, i:i+1] = denormalization(data[:, i:i+1],
                                                 mean=self.means_dim[i], std=self.std_dim[i])

            return data

    def normalize(self, data: torch.tensor, feature: str):
        if feature == "location":
            for i in range(data.shape[1]):
                data[:, i:i+1] = normalization(data[:, i:i+1],
                                               mean=self.means_loc[i], std=self.std_loc[i])
            return data
        if feature == "dimension":
            for i in range(data.shape[1]):
                data[:, i:i+1] = normalization(data[:, i:i+1],
                                               mean=self.means_dim[i], std=self.std_dim[i])
            return data
