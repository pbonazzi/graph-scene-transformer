from pyexpat import model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import dgl
from network.layers.multi_head_attention import MultiHeadAttentionLayer
from network.layers.graph_convolution import GraphConvolution


class TransformerEncoder(nn.Module):
    """ Transformer Encoder

    Author : Vijay Dwivedi
    https://github.com/graphdeeplearning/graphtransformer

    Modified by: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    def __init__(self, device: torch.device, in_dim: int, out_dim: int, num_heads: int, dropout: float = 0.0, layer_norm: bool = False,
                 batch_norm: bool = True, residual: bool = True, use_bias: bool = False, name: str = "transformer",
                 type_of_attention: str = "classic", layers_gcn: int = 2):
        super().__init__()

        """ Transformer encoder layer 
        """

        self.name = name
        self.device = device

        # load inputs parameters
        self.type_of_attention = type_of_attention
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # initialize multi head attention layer
        self.attention = MultiHeadAttentionLayer(in_dim=in_dim,
                                                 out_dim=out_dim // num_heads,
                                                 num_heads=num_heads,
                                                 use_bias=use_bias,
                                                 name='multi_head_attention')

        # linear
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        # Normalizations before FFN
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        # Normalizations before FFN
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        if self.type_of_attention == "graphormer":
            # Graph Convolution for h and e
            self.gcn_encoder = nn.ModuleList([GraphConvolution(
                layer1=[out_dim * 3,
                        out_dim * 4, out_dim * 3],
                name="convolution_layer" + str(i),
                device=self.device) for i in range(layers_gcn)])

        else:
            # Feed Forward Network for h
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
            self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

            # FFN for edges
            self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
            self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        # Normalizations after FFN
        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        # Normalizations after FFN
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g: dgl.DGLGraph, h: torch.tensor, e: torch.tensor, padding_mask: np.ndarray = None,
                output_mask: np.ndarray = None):
        """
        """

        # for first residual connection
        h_in1 = h
        e_in1 = e

        # multi-head attention output
        g_out, h_attn_out, e_attn_out = self.attention(g=g, h=h, e=e,
                                                       padding_mask=padding_mask,
                                                       output_mask=output_mask,
                                                       type_of_attention=self.type_of_attention)

        # reshape
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        # randomly 0 some with probability self.dropout if training
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        # linear
        h = self.O_h(h)
        e = self.O_e(e)

        # residual connection
        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        # Normalizations before FFN
        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        # Normalizations before FFN
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        # for second residual connection
        h_in2 = h
        e_in2 = e

        if self.type_of_attention == "graphormer":
            for gcn_layer in self.gcn_encoder:
                h, e = gcn_layer(
                    g, h, e)
        else:
            # FFN for h
            h = self.FFN_h_layer1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.FFN_h_layer2(h)

            # FFN for e
            e = self.FFN_e_layer1(e)
            e = F.relu(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer2(e)

        # Residual connection
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        # Normalizations after FFN
        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        # Normalizations after FFN
        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return g_out, h, e

    def __repr__(self):

        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
