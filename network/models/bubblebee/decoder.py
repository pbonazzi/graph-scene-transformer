#
# Author: Pietro Bonazzi, Mengqi Wang
#

# import libraries
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest

# import custom methods
from data.utils import get_category, get_location, get_orientation, get_dimension
from network.layers.multi_layer_perceptron import build_mlp, simple_relu
from network.layers.transformer_decoder import TransformerDecoder
from network.layers.graph_convolution import GraphConvolution


class Decoder(nn.Module):

    def __init__(self, params: dict):
        super().__init__()

        # decoding embedding
        self.name = "decoding container"
        self.autoregressive = False
        self.device = params['device']
        self.hidden_dim = params['decoder']['hidden_dimension']

        # node embedding
        self.nn_category = nn.Embedding(params['data']['n_of_objects'], params['data']["enc_dim_object"])

        # layout
        self.char_floorplan =  params["encoder"]["floorplan"]
        self.char_positional_encoding =  params["data"]["positional_encoding"]
        if self.char_positional_encoding == "laplacian":
            self.nn_pos_enc = nn.Linear( params["data"]["positional_encoding_dim"], params['decoder']['hidden_dimension'])
        elif self.char_positional_encoding == "wl":
            self.nn_pos_enc = nn.Embedding(params["data"]["positional_encoding_max_graph"], params['decoder']['hidden_dimension'])
        else:
            pass

        # node embedding
        self.nn_node = nn.Linear(params['encoder']['hidden_dimension']+params['data']["enc_dim_object"], params['encoder']['hidden_dimension'])

        # edge embedding
        self.nn_edges = nn.Embedding(params['data']['n_of_relationships'] + 1, params['decoder']['hidden_dimension'])

        # graph convolution
        self.bool_use_gcn = False
        if params['decoder']["n_of_convolution_layers"] > 0:
            self.bool_use_gcn = True
            self.gcn_decoder = nn.ModuleList([GraphConvolution(
                layer1=[self.hidden_dim * 3,
                        self.hidden_dim * 4, self.hidden_dim * 3],
                name="convolution_layer" + str(i),
                device=self.device) for i in range(params_dec['n_of_convolution_layers'])])

        # type of attention
        self.bool_use_transformer = False
        if params['decoder']["n_of_transformers_layers"] > 0:
            self.bool_use_transformer = True
            self.transformer_decoder = nn.ModuleList([TransformerDecoder(device=self.device,
                                                                         in_dim=self.hidden_dim,
                                                                         out_dim=self.hidden_dim,
                                                                         num_heads=params['decoder']['n_of_attention_heads'],
                                                                         dropout=params['dropout'],
                                                                         layer_norm=params['decoder']['layer_norm'],
                                                                         batch_norm=params['decoder']['batch_norm'],
                                                                         residual=params['decoder']['residual'],
                                                                         type_of_attention=params['decoder']['type_of_attention'],
                                                                         layers_gcn=params['decoder']["n_of_convolution_layers"],
                                                                         name="transformer" + str(i))
                                                      for i in range(params['decoder']['n_of_transformers_layers'])])

        self.dim_net = build_mlp([self.hidden_dim, int(self.hidden_dim / 4), 3],
                                 batch_norm='batch',
                                 final_nonlinearity=False)
        self.loc_net = build_mlp([self.hidden_dim, int(self.hidden_dim / 4), 3],
                                 batch_norm='batch',
                                 final_nonlinearity=False)
        self.angle_net = build_mlp([self.hidden_dim, int(self.hidden_dim / 4), params['data']['n_of_orientation_bins']],
                                   batch_norm='batch',
                                   final_nonlinearity=False)


    def forward(self, z_node: torch.tensor, z_edge: torch.tensor, graph: dgl.DGLGraph, ids: list, layout: torch.tensor, user_mode: bool = False):
        """ Return scores and targets
        """
        dgl_graph = graph.to(self.device).local_var()

        # add special nodes layout
        if self.use_floor_plan:
            ub_graphs_dc = dgl.unbatch(dgl_graph)
            for i in range(len(ids)):
                ub_graphs_dc[i].add_nodes(
                    1, {"global_id": torch.tensor([0]).to(self.device)})
                for j in range(ub_graphs_dc[i].number_of_nodes() - 1):
                    ub_graphs_dc[i].add_edges(ub_graphs_dc[i].number_of_nodes() - 1, j,
                                              {"feat": torch.tensor([0]).to(self.device)})
                assert ub_graphs_dc[i].number_of_edges() > 0, print(
                    f'Graph has no edges: {ub_graphs_dc[i]}')
            dgl_graph = dgl.batch(ub_graphs_dc)

        # node encoder
        category = get_category(graph=dgl_graph, device=self.device, ohc=False)
        m_node = self.nn_category(category)

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

        # edge encoder
        m_edge = dgl_graph.edata['feat']
        m_edge = self.input_edges_embed(m_edge)

        # concatenate noise
        if self.char_floorplan != 'none':
            padd = torch.zeros(
                (m_node.shape[0]-z_node.shape[0], z_node.shape[1])).to(self.device)
            z_node_pad = torch.cat([z_node, padd], dim=0)
            padd = torch.zeros(
                (m_edge.shape[0]-z_edge.shape[0], z_edge.shape[1])).to(self.device)
            z_edge_pad = torch.cat([z_edge, padd], dim=0)
            m_node_with_noise_dc = torch.cat(
                [m_node, z_node_pad], dim=1)
            m_edge_with_noise_dc = torch.cat(
                [m_edge, z_edge_pad], dim=1)
        else:
            m_node_with_noise_dc = torch.cat(
                [m_node, z_node], dim=1)
            m_edge_with_noise_dc = torch.cat(
                [edge_embedding, z_edge], dim=1)


        m_node_dc = simple_relu(m_node_with_noise_dc)

        m_edge_dc = simple_relu(m_edge_with_noise_dc)

        # floor plan decoder
        if self.char_floorplan != 'none':
            idx_floor_plan = (dgl_graph.ndata["global_id"] == 0).nonzero(
                as_tuple=True)[0]
            for i, idx in enumerate(idx_floor_plan):
                m_node_dc[idx] = layout[i].clone()

        # mix graph convolution and transformers
        mixed_gnn_attention = False
        if mixed_gnn_attention:
            for gcn_layer, transformer_layer in zip_longest(self.gcn_decoder, self.transformer_decoder, fillvalue='?'):
                if gcn_layer != '?':
                    m_node_dc, m_edge_dc = gcn_layer(
                        dgl_graph, m_node_dc, m_edge_dc)
                if transformer_layer != '?':
                    dgl_graph, m_node_dc, m_edge_dc = transformer_layer(g=dgl_graph, h=m_node_dc,e=m_edge_dc)
        else:
            # graph convolution
            if self.use_graph_conv:
                for gcn_layer in self.gcn_decoder:
                    m_node_dc, m_edge_dc = gcn_layer(
                        dgl_graph, m_node_dc, m_edge_dc)

            # transformer encoder
            if self.use_transformer:
                for transformer_layer in self.transformer_decoder:
                    dgl_graph, m_node_dc, m_edge_dc = transformer_layer(g=dgl_graph, h=m_node_dc,
                                                                                       e=m_edge_dc)

        # remove floor plan node
        if self.use_floor_plan:
            ub_graphs_dc = dgl.unbatch(dgl_graph)
            node_counter, edge_counter = 0, 0

            for graph in ub_graphs_dc:
                idx_fp_nodes = (graph.ndata["global_id"] == 0).nonzero(
                    as_tuple=True)[0]
                idx_fp_edges = (graph.edata["feat"] == 0).nonzero(
                    as_tuple=True)[0]
                assert idx_fp_nodes.size(dim=0) == 1
                assert idx_fp_nodes.item() == graph.number_of_nodes()-1
                assert idx_fp_edges.size(dim=0) == graph.number_of_nodes()-1
                assert torch.eq(idx_fp_edges, torch.arange(graph.number_of_edges(
                )-graph.number_of_nodes()+1, graph.number_of_edges()).to(self.device)).all()

                m_node_dc_g = torch.index_select(m_node_dc, 0, torch.arange(
                    node_counter, node_counter+graph.number_of_nodes()-1).to(self.device))
                m_edge_dc_g = torch.index_select(m_edge_dc, 0, torch.arange(
                    edge_counter, edge_counter+graph.number_of_edges()-graph.number_of_nodes()+1).to(self.device))

                if node_counter == 0:
                    temp_m_node_dc = m_node_dc_g
                else:
                    temp_m_node_dc = torch.cat(
                        [temp_m_node_dc, m_node_dc_g], dim=0)

                if edge_counter == 0:
                    temp_m_edge_dc = m_edge_dc_g
                else:
                    temp_m_edge_dc = torch.cat(
                        [temp_m_edge_dc, m_edge_dc_g], dim=0)
                node_counter = node_counter + graph.number_of_nodes()
                edge_counter = edge_counter + graph.number_of_edges()
                graph.remove_nodes(idx_fp_nodes.item())

            dgl_graph = dgl.batch(ub_graphs_dc)
            m_node_dc = temp_m_node_dc
            m_edge_dc = temp_m_edge_dc

        # scores
        angles_pred = self.angle_net(m_node_dc)
        dim_pred = self.dim_net(m_node_dc)
        loc_pred = self.loc_net(m_node_dc)
        scores = torch.cat([dim_pred, loc_pred, angles_pred], dim=1)

        # targets
        if "orientation" in list(dgl_graph.ndata.keys()) and user_mode == False:
            category_vec = get_category(
                graph=dgl_graph, device=self.device, ohc=True)
            orientation_vec = get_orientation(
                graph=dgl_graph, device=self.device, ohc=True)
            location = get_location(
                graph=dgl_graph, decomposed=False).to(self.device)
            dimension = get_dimension(
                graph=dgl_graph, decomposed=False).to(self.device)
            targets = torch.cat(
                [dimension, location, orientation_vec], dim=-1).to(self.device)
        else:
            targets = 0

        return dgl_graph, scores, targets
