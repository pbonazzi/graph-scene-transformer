# import libraries
import dgl
import torch
import torch.nn as nn

# import custom methods
from network.layers.multi_layer_perceptron import build_mlp


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphConvolution(nn.Module):
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    def __init__(self, device: object, name: str, layer1: list, layer2: list = None):
        super().__init__()
        self.dim1 = layer1
        self.name = name
        self.device = device
        self.graph_conv1 = build_mlp(
            layer1, batch_norm='batch', final_nonlinearity=False)
        self.graph_conv1.apply(_init_weights)
        self.use_layer2 = False
        if layer2 is not None:
            self.use_layer2 = True
            self.graph_conv2 = build_mlp(
                layer2, batch_norm='none', final_nonlinearity=False)
            self.graph_conv2.apply(_init_weights)

    def forward(self, graph: dgl.DGLGraph, node_embedding: torch.tensor, edge_embedding: torch.tensor):

        # bootstrapping
        graph = graph.to(self.device).local_var()

        src_nodes = graph.edges()[0]
        dst_nodes = graph.edges()[1]

        triplet_embed = torch.cat([node_embedding[src_nodes], edge_embedding,
                                   node_embedding[dst_nodes]], dim=-1).to(self.device)
        new_triplet_embed = self.graph_conv1(triplet_embed)

        dim = int(self.dim1[-1] / 3)

        src_embed = new_triplet_embed[:, :dim]
        edg_embed = new_triplet_embed[:, dim:2 * dim]
        dst_embed = new_triplet_embed[:, 2 * dim:]

        new_node_embed = torch.zeros(
            graph.number_of_nodes(), dim).to(self.device)

        s_idx_exp = src_nodes.view(-1, 1).expand_as(src_embed)
        o_idx_exp = dst_nodes.view(-1, 1).expand_as(dst_embed)

        new_node_embed = new_node_embed.scatter_add(0, s_idx_exp, src_embed)

        new_node_embed = new_node_embed.scatter_add(0, o_idx_exp, dst_embed)

        if self.use_layer2:
            new_node_embed = self.graph_conv2(new_node_embed)

        return new_node_embed, edg_embed
