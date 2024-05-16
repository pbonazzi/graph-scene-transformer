import dgl
import torch.nn as nn
import dgl.function as fn
from network.layers.utils import *


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, use_bias: bool, name: str):
        super().__init__()

        """ Function called by class GraphTransformerLayer in layers/edges_transformer_layer.py. 

            Parameters
            ----------
            - int in_dim : embedding of the inputs 
            - int out_dim : desired size of the output
            - int num_heads : number of head attention layers
            - float dropout : list scan ids, used by the resnet
            - bool use_bias : learnable additive bias
            - string name : name of the layer
            
            Author : Vijay Dwivedi
            https://github.com/graphdeeplearning/graphtransformer

        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.name = name

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Q_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.Q_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g: dgl.DGLGraph):
        """ Edge based attention . Summing the scores and not applying the edge feature
        """
        list_of_graphs = dgl.unbatch(g)
        for g_ in list_of_graphs:
            # Compute attention score
            g_.apply_edges(src_dot_dst_sum('K_h', 'Q_h', 'score'))
            g_.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
            # Update node state
            g_.send_and_recv(g_.edges(), fn.src_mul_edge('V_h', 'score', 'V_h'),
                             fn.sum('V_h', 'wV'))
            g_.send_and_recv(g_.edges(), fn.copy_edge('score', 'score'),
                             fn.sum('score', 'z'))
        return dgl.batch(list_of_graphs)

    def propagate_attention_working_with_nodes(self, g: dgl.DGLGraph, padding_mask: np.ndarray, output_mask: np.ndarray):
        """ Calculate attention as in vanilla transformers. Option to sum the edge information as a node feature.

            Author: Pietro Bonazzi
            https://github.com/uzh-rpg/scene_graph_3d

        """

        final_list = dgl.unbatch(g)
        list_of_graphs = dgl.unbatch(g)

        for i in range(len(list_of_graphs)):
            list_of_graphs[i].apply_nodes(src_dot_dst('K_h', 'Q_h', "nodes"))

            list_of_graphs[i].apply_nodes(
                scaling(num_heads=self.num_heads, scale_constant=np.sqrt(self.out_dim), name="nodes"))

            if padding_mask is not None:
                list_of_graphs[i].apply_nodes(
                    masking(num_heads=self.num_heads, keys=padding_mask, name="nodes"))

            if output_mask is not None:
                if np.sum(output_mask) != 0:
                    list_of_graphs[i].apply_nodes(
                        masking(num_heads=self.num_heads, keys=output_mask, name="nodes"))

            list_of_graphs[i].apply_nodes(softmax(self.num_heads))

            list_of_graphs[i].apply_nodes(apply_attention(self.num_heads))
        for i in range(len(list_of_graphs)):
            final_list[i].ndata["h_out"] = list_of_graphs[i].ndata["h_out"]

        return dgl.batch(final_list)

    def propagate_attention_as_a_generalization_of_graph_to_transformer(self, g: dgl.DGLGraph, aggregate_on_dst: bool):
        """ Graph Transformer , Edge-Based Attention
            Modified so that only nodes in the same scenes are affecting one another. 

            Author: Pietro Bonazzi
            https://github.com/uzh-rpg/scene_graph_3d

        """
        final_list = dgl.unbatch(g)
        list_of_graphs = dgl.unbatch(g)

        for i in range(len(list_of_graphs)):

            list_of_graphs[i].apply_edges(
                src_multiply_dst('K_h', 'Q_h', 'score'))  # , edges)

            list_of_graphs[i].apply_edges(
                scaling_multiply('score', np.sqrt(self.out_dim)))

            list_of_graphs[i].apply_edges(imp_exp_attn('score', 'Q_e'))

            list_of_graphs[i].apply_edges(out_edge_features('score'))

            list_of_graphs[i].apply_edges(exp('score'))

            eids = list_of_graphs[i].edges()

            if aggregate_on_dst:
                # destination aggregation
                list_of_graphs[i].apply_edges(
                    dst_edge_node('V_h', 'V_h'))

                list_of_graphs[i].apply_edges(
                    imp_exp_attn('V_h', 'score'))

                list_of_graphs[i].send_and_recv(eids, fn.copy_edge(
                    'V_h', 'V_h'), fn.sum('V_h', 'wV'))
            else:
                list_of_graphs[i].send_and_recv(eids, fn.src_mul_edge(
                    'V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

            list_of_graphs[i].send_and_recv(eids, fn.copy_edge(
                'score', 'score'), fn.sum('score', 'z'))

            final_list[i].ndata["wV"] = list_of_graphs[i].ndata["wV"].clone()
            final_list[i].ndata["z"] = list_of_graphs[i].ndata["z"].clone()
            final_list[i].edata["e_out"] = list_of_graphs[i].edata["e_out"].clone()

        return dgl.batch(final_list)

    def propagate_attention_aggregate(self, g: dgl.DGLGraph, padding_mask, output_mask):

        # g_node = self.propagate_attention(g)
        g_node = self.propagate_attention_working_with_nodes(
            g, padding_mask, output_mask, None)
        h_out_node = g_node.ndata["h_out"]
        e_out_node = g_node.edata['Q_e']

        g_edge = self.propagate_attention_as_a_generalization_of_graph_to_transformer(
            g, aggregate_on_dst=False)
        h_out_edge = g_edge.ndata['wV'] / \
            (g_edge.ndata['z'] + torch.full_like(g_edge.ndata['z'], 1e-6))
        e_out_edge = g_edge.edata['e_out']

        h_out = h_out_node + h_out_edge
        e_out = e_out_node + e_out_edge

        return g, h_out, e_out

    def forward(self, g: dgl.DGLGraph, h: torch.tensor, e: torch.tensor, padding_mask: np.ndarray = None,
                output_mask: np.ndarray = None, type_of_attention: str = "node_attention_use_edg"):
        """ Computes the attentions scores for edges and nodes

            Parameters
            ----------
            - dgl.DGLHeteroGraph g
            - torch.Tensor h : (torch.Tensor([55*n, out_dim])
            - torch.Tensor e : (torch.Tensor([g.number_edges(), out_dim])
            - node (string) : node identifier.

            Return
            ----------
            - torch.Tensor h_out : (torch.Tensor([55*n, num_heads, num_heads])
            - torch.Tensor e_out : (torch.Tensor([g.number_edges(), num_heads, num_heads])


            Author: Pietro Bonazzi
            https://github.com/uzh-rpg/scene_graph_3d


        """
        # src_field
        Q_h = self.Q(h)
        # reshape (i.e. 55, 8, 64)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)

        # dst_field
        K_h = self.K(h)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)

        # value field
        V_h = self.V(h)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        # available edge features
        Q_e = self.Q_e(e)

        g.edata['Q_e'] = Q_e.view(-1, self.num_heads, self.out_dim)

        # compute attention scores
        if type_of_attention in ["classic", "graphormer"]:
            g = self.propagate_attention_working_with_nodes(
                g, padding_mask, output_mask)
            h_out = g.ndata['h_out']
            g.edata['e_out'] = g.edata['Q_e']

        elif type_of_attention == "dgl_tutorial":
            g = self.propagate_attention(g)
            h_out = g.ndata['wV'] / \
                (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
            g.edata['e_out'] = g.edata['Q_e']

        elif type_of_attention == "as_in_generalization_graph_to_transformer":
            g = self.propagate_attention_as_a_generalization_of_graph_to_transformer(
                g, aggregate_on_dst=False)
            h_out = g.ndata['wV'] / \
                (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        elif type_of_attention == "as_in_generalization_graph_to_transformer_on_dst":
            g = self.propagate_attention_as_a_generalization_of_graph_to_transformer(
                g, aggregate_on_dst=True)
            h_out = g.ndata['wV'] / \
                (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        elif type_of_attention == "node_edge_combined":
            g, h_out, e_out = self.propagate_attention_aggregate(
                g, padding_mask, output_mask)
            return g, h_out, e_out
        else:
            raise ValueError("You need to select one of the attention methods")

        e_out = g.edata['e_out']

        return g, h_out, e_out
