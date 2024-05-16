#
# Author: Pietro Bonazzi, Mengqi Wang
#

# import libraries
import pdb
import dgl
import torch
import torch.nn as nn
from itertools import zip_longest

# import custom methods
from data.utils import get_category, get_location, get_orientation, get_dimension
from network.layers.transformer_encoder import TransformerEncoder
from network.layers.multi_layer_perceptron import build_mlp, simple_relu
from network.layers.graph_convolution import GraphConvolution


class Encoder(nn.Module):

    def __init__(self, params: dict):
        super().__init__()
        self.name: str = "ENCODER"

        self.nn_category = nn.Embedding(params['data']['n_of_objects'], params['data']["enc_dim_object"])
        self.nn_orientation = nn.Embedding(params['data']['n_of_orientation_bins'], params['data']["enc_dim_rotation_z"])
        self.nn_size = nn.Linear(3, params['data']["enc_dim_size"])
        self.nn_translation_ = nn.Linear(3, params['data']["enc_dim_transln"])
        uint_concat = params['data']["enc_dim_object"] + params['data']["enc_dim_rotation_z"] + \
                params['data']["enc_dim_size"] +  params['data']["enc_dim_transln"]

        self.nn_node = nn.Linear(uint_concat, params['encoder']['hidden_dimension'])
        self.nn_edges = nn.Embedding(params['data']['n_of_relationships'] + 1, params['encoder']['hidden_dimension'])

        self.char_floorplan =  params["encoder"]["floorplan"]
        self.char_positional_encoding =  params["data"]["positional_encoding"]
        if self.char_positional_encoding == "laplacian":
            self.nn_pos_enc = nn.Linear( params["data"]["positional_encoding_dim"], uint_concat)
        elif self.char_positional_encoding == "wl":
            self.nn_pos_enc = nn.Embedding(params["data"]["positional_encoding_max_graph"], uint_concat)
        else:
            pass

        self.nn_feat_dropout = nn.Dropout(params['in_feat_dropout'])

        # graph convolution
        self.bool_use_gcn = False
        if params['encoder']["n_of_convolution_layers"] > 0:
            self.bool_use_gcn = True
            self.nn_gcn = nn.ModuleList([GraphConvolution(
                layer1=[params['encoder']['hidden_dimension']* 3,
                        params['encoder']['hidden_dimension'] * 4, 
                        params['encoder']['hidden_dimension'] * 3],
                name="convolution_layer" + str(i),
                device=params['device']) for i in range(params['encoder']["n_of_convolution_layers"])])

        # encoder
        self.bool_use_transformer = False
        if params['encoder']["n_of_transformers_layers"] > 0:
            self.bool_use_transformer = True
            self.nn_transformer = nn.ModuleList([TransformerEncoder(device=params['device'],
                                                                         in_dim=params['encoder']['hidden_dimension'],
                                                                         out_dim=params['encoder']['hidden_dimension'],
                                                                         num_heads=params['encoder']['n_of_attention_heads'],
                                                                         dropout=params['dropout'],
                                                                         layer_norm=params['encoder']['layer_norm'],
                                                                         batch_norm=params['encoder']['batch_norm'],
                                                                         residual=params['encoder']['residual'],
                                                                         type_of_attention=params['encoder']["type_of_attention"],
                                                                         layers_gcn=params['encoder']["n_of_convolution_layers"],
                                                                         name="transformer" + str(i))
                                                      for i in range(params['encoder']['n_of_transformers_layers'])])

        # latent space
        self.bool_use_latentspace = False
        if params['latent'] != 'none':
            self.bool_use_latentspace = True
            self.nn_mean_var_node = build_mlp([ params['encoder']['hidden_dimension'], 
                                                params['encoder']['hidden_dimension']* 2, 
                                                params['encoder']['hidden_dimension'] * 3], batch_norm='batch')
            
            self.nn_mean_var_edge = build_mlp([ params['encoder']['hidden_dimension'], 
                                                params['encoder']['hidden_dimension']* 2, 
                                                params['encoder']['hidden_dimension'] * 3], batch_norm='batch')
                                                
            self.nn_mean_node = build_mlp([ params['encoder']['hidden_dimension']*3, 
                                            params['encoder']['hidden_dimension']], 
                                            final_nonlinearity=False,
                                            batch_norm='batch')
            self.nn_mean_edge = build_mlp([ params['encoder']['hidden_dimension']*3, 
                                            params['encoder']['hidden_dimension']], 
                                            final_nonlinearity=False,
                                            batch_norm='batch')
            self.nn_var_node = build_mlp([ params['encoder']['hidden_dimension']*3, 
                                            params['encoder']['hidden_dimension']], 
                                            final_nonlinearity=False,
                                            batch_norm='batch')
            self.nn_var_edge = build_mlp([ params['encoder']['hidden_dimension']*3, 
                                            params['encoder']['hidden_dimension']], 
                                            final_nonlinearity=False,
                                            batch_norm='batch')

            self.torch_device : torch.device = params['device']

    def forward(self, dgl_graph: dgl.DGLGraph, ids: list, layout: torch.tensor, deterministic: bool):
        """ Return scores and targets
        """

        dgl_graph = dgl_graph.to(self.torch_device).local_var()


        if self.char_floorplan != 'none':
            dgl_unbatched = dgl.unbatch(dgl_graph)
            for i in range(len(ids)):
                # add special node 'floor plan'
                dgl_unbatched[i].add_nodes(
                    1, {"global_id": torch.tensor([0]).to(self.torch_device)})

                for j in range(dgl_unbatched[i].number_of_nodes() - 1):
                    dgl_unbatched[i].add_edges(dgl_unbatched[i].number_of_nodes() - 1, j,
                                              {"feat": torch.tensor([0]).to(self.torch_device)})

            dgl_graph = dgl.batch(dgl_unbatched)


        
        location = get_location(graph=dgl_graph, decomposed=False).to(self.torch_device)
        dimension = get_dimension(graph=dgl_graph, decomposed=False)
        rotation_z = get_orientation(graph=dgl_graph, device=self.torch_device, ohc=False)
        category = get_category(graph=dgl_graph, device=self.torch_device, ohc=False)

        m_location : torch.tensor = self.nn_translation_(location)
        m_category : torch.tensor = self.nn_category(category)
        m_rotation_z : torch.tensor = self.nn_orientation(rotation_z)
        m_dimension : torch.tensor = self.nn_size(dimension)


        m_edge = simple_relu(self.nn_edges(dgl_graph.edata['feat']).long())
        m_node = torch.cat([m_category, m_rotation_z, m_dimension, m_location], dim=-1).to( self.torch_device)


        if self.char_floorplan == 'laplacian':
            m_lap_pos_enc = dgl_graph.ndata['lap_pos_enc']

            m_sign_flip = torch.rand(m_lap_pos_enc.size(1)).to(self.torch_device)
            m_sign_flip[m_sign_flip >= 0.5] = 1.0
            m_sign_flip[m_sign_flip < 0.5] = -1.0
            m_lap_pos_enc = m_lap_pos_enc * m_sign_flip.unsqueeze(0)
            
            m_lap_pos_enc = self.nn_pos_enc(m_lap_pos_enc.float())
            m_node = m_node.clone() + m_lap_pos_enc

        elif self.char_floorplan == 'wl':
            m_wl_pos_enc = dgl_graph.ndata['wl_pos_enc']
            m_wl_pos_enc = self.nn_pos_enc(m_wl_pos_enc)
            m_node = m_node.clone() + m_wl_pos_enc
        else:
            pass


        m_node = simple_relu(self.nn_node(m_node.float()))

        # floor plan encoder
        if self.char_floorplan != 'none':
            list_idx_floor_plan = (dgl_graph.ndata["global_id"] == 0).nonzero(as_tuple=True)[0]
            for i, idx in enumerate(list_idx_floor_plan):
                m_node[idx] = layout[i].clone()

        # mix graph convolution and transformers
        mixed_gnn_attention = False
        if mixed_gnn_attention:
            for gcn_layer, transformer_layer in zip_longest(self.nn_gcn, self.nn_transformer, fillvalue='?'):
                if gcn_layer != '?':
                    m_node, m_edge = gcn_layer(dgl_graph, m_node, m_edge)
                if transformer_layer != '?':
                    dgl_graph, m_node, m_edge = transformer_layer(g=dgl_graph, h=m_node,e=m_edge)
        else:
            # graph convolution
            if self.bool_use_gcn:
                for gcn_layer in self.nn_gcn:
                    m_node, m_edge = gcn_layer(dgl_graph, m_node, m_edge)

            # transformer encoder
            if self.bool_use_transformer:
                for transformer_layer in self.nn_transformer:
                    dgl_graph, m_node, m_edge = transformer_layer(g=dgl_graph, h=m_node, e=m_edge)


        # remove floor plan nodes and embeddings
        if self.char_floorplan != 'none':
            idx_fp_nodes = (dgl_graph.ndata["global_id"]== 0).nonzero(as_tuple=True)[0]
            idx_fp_edges = (dgl_graph.edata["feat"] == 0).nonzero(as_tuple=True)[0]

            for i in range(len(idx_fp_nodes) - 1, -1, -1):
                m_node = m_node[torch.arange(m_node.size(0)).to(self.torch_device) != idx_fp_nodes[i]].to(self.torch_device)

            for i in range(len(idx_fp_edges) - 1, -1, -1):
                m_edge = m_edge[torch.arange(m_edge.size(0)).to(self.torch_device) != idx_fp_edges[i]].to(self.torch_device)

            ub_graphs_ec = dgl.unbatch(dgl_graph)
            for graph in ub_graphs_ec:
                idx_fp_nodes = (graph.ndata["global_id"] == 0).nonzero(as_tuple=True)[0]
                graph.remove_nodes(idx_fp_nodes.item())

            dgl_graph = dgl.batch(ub_graphs_ec)

        # mean_var: weight sharing of mean and var
        if self.bool_use_latentspace:
            return m_node, m_edge, 0, 0
        else:
            obj_vecs_3d = self.nn_mean_var_node(m_node)
            edge_vecs_3d = self.nn_mean_var_edge(m_edge)

            mu_node = self.nn_mean_node(obj_vecs_3d)
            logvar_node = self.nn_var_node(obj_vecs_3d)

            mu_edge = self.nn_mean_edge(edge_vecs_3d)
            logvar_edge = self.nn_var_edge(edge_vecs_3d)

            return mu_node, logvar_node, mu_edge, logvar_edge
