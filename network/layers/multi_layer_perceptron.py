import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


def simple_relu(x):
    return torch.maximum(x, torch.zeros_like(x))

# class MLPReadout(nn.Module):

#     def __init__(self, input_dim, output_dim, L=2):
#         super().__init__()

#         # hidden layers
#         list_fc_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]

#         # output layer
#         list_fc_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))

#         self.FC_layers = nn.ModuleList(list_fc_layers)
#         self.L = L

#     def forward(self, y):

#         # hidden layers
#         for l in range(self.L):
#             y = self.FC_layers[l](y)
#             y = F.leaky_relu(y)

#         # output layer
#         y = self.FC_layers[self.L](y)

#         return y
