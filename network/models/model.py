# libraries
import os, pdb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import dgl, pdb
import torchvision.models as models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# custom methods
from config.paths import THREED_SSG_PLUS
from network.models.bubblebee.latent import Gaussian
from network.models.bubblebee.encoder import Encoder
from network.layers.floor_plan import layout_encoder
from network.models.bubblebee.decoder import Decoder
from network.metrics.multilayerloss import MultiLossLayer

from data.utils import get_category, get_vocabulary

class BubbleBee(nn.Module):

    def __init__(self, params: dict):
        super().__init__()
        
        # architecture
        self.nn_encoder : nn.Module = Encoder(params=params)
        self.nn_decoder : nn.Module = Decoder(params=params)


        self.torch_device : torch.device = params['device']
        self.str_root : str = params['root']
        self.int_hidden_dimension : int = params['encoder']['hidden_dimension']


        self.char_latent_method = params['latent']
        if self.char_latent_method != 'none':
            self.mu_node, self.logvar_node, self.mu_edge, self.logvar_edge = None, None, None, None
        else:
            self.nn_latent_space : nn.Module = Gaussian()


        if params['loss']['multi_layer']:
            self.nn_multi_loss_layer : nn.Module = MultiLossLayer(
                ["cat", "cat", "con"])

        self.char_floorplan: str = params['encoder']['floorplan']
        self.layout = None
        if self.char_floorplan != 'none':
            self.nn_resnet50 = models.resnext50_32x4d(pretrained=True)
            self.nn_resnet50.eval()
            self.nn_resnet50.fc = nn.Linear(
                self.nn_resnet50.fc.in_features, 
                params["encoder"]['hidden_dimension'])

    
    def forward(self, graph: dgl.DGLGraph, ids: list):

        graph_ec : dgl.Graph = graph.to(self.torch_device).local_var()
        
        if self.char_floorplan != 'none':
            self.layout = layout_encoder(
                ids, self.resnet50, self.torch_device, self.char_floorplan)

        if self.char_latent_method != 'none':
            node_embedding_ec, edge_embedding_ec, _, _ = self.nn_encoder(
                graph_ec, ids, self.layout, deterministic=self.char_latent_method)

            new_graph, scores, targets = self.nn_decoder(
                node_embedding_ec, edge_embedding_ec, graph, ids, self.layout)
        
        else:
            self.mu_node, self.logvar_node, self.mu_edge, self.logvar_edge = self.nn_encoder(
                graph_ec, ids, self.layout, deterministic=self.char_latent_method)

            z_node, z_edge = self.latent_space(
                self.mu_node, self.logvar_node, self.mu_edge, self.logvar_edge)

            new_graph, scores, targets = self.nn_decoder(
                z_node, z_edge, graph, ids, self.layout)

        return new_graph, scores, targets, self.mu_node, self.logvar_node, self.mu_edge, self.logvar_edge

    def inference(self, graphs: dgl.DGLGraph, ids: list, posterior_statistics=None, var_scale=1, user_mode=False):

        # bootstrapping
        graph = graphs.to(self.torch_device).local_var()

        if self.char_floorplan != 'none':
            self.layout = layout_encoder(
                ids, self.resnet50, self.torch_device, self.char_floorplan)

        # node encoder
        category_vec = get_category(graph=graph, device=self.torch_device, ohc=False)
        category_embed = self.nn_encoder.nn_category(category_vec).to(self.torch_device)

        # edge encoder
        edge_label = graph.edata['feat'].to(self.torch_device)
        edge_embedding = self.nn_encoder.nn_edges(edge_label).to(self.torch_device)

        # compute statistics
        node_cnt = category_embed.size(0)
        edge_cnt = edge_embedding.size(0)
        ec_hidd_dim = self.int_hidden_dimension

        if posterior_statistics is None:
            z_node = torch.from_numpy(
                np.random.multivariate_normal(np.zeros(ec_hidd_dim), np.eye(ec_hidd_dim), node_cnt)).float().to(self.torch_device)
            z_edge = torch.from_numpy(
                np.random.multivariate_normal(np.zeros(ec_hidd_dim), np.eye(ec_hidd_dim), edge_cnt)).float().to(self.torch_device)
        else:
            mean_node_est, cov_node_est, mean_edge_est, cov_edge_est = posterior_statistics
            z_node = torch.from_numpy(np.random.multivariate_normal(mean_node_est, var_scale * cov_node_est, node_cnt)).float().to(
                self.torch_device)
            z_edge = torch.from_numpy(np.random.multivariate_normal(mean_edge_est, var_scale * cov_edge_est, edge_cnt)).float().to(
                self.torch_device)

        # decoder
        new_graph, scores, targets = self.nn_decoder(
            z_node, z_edge, graph, ids, self.layout, user_mode=user_mode)

        return new_graph, scores, targets

    def collect_train_statistics(self, train_loader, plot):
        """ adapted from 
        https://github.com/he-dhamo/graphto3d/blob/main/dataset/dataset.py

        """
        mean_node_cat, mean_edge_cat = None, None
        z_node_cat, z_edge_cat, category_vec_cat = None, None, None

        for idx, (batch_graphs, ids) in tqdm(enumerate(train_loader)):

            # bootstrapping
            graph = batch_graphs.to(self.torch_device).local_var()

            if self.char_floorplan != 'none':
                self.layout = layout_encoder(
                    ids, self.resnet50, self.torch_device, self.char_floorplan)

            # encoder
            mu_node, logvar_node, mu_edge, logvar_edge = self.nn_encoder(
                graph, ids, deterministic=self.char_latent_method, layout=self.layout)

            if plot:
                z_node, z_edge = self.latent_space(
                    mu_node, logvar_node, mu_edge, logvar_edge)
                category_vec = get_category(
                    graph=graph, device=self.torch_device, ohc=False)

            mean_node, logvar_node, mean_edge, logvar_edge = mu_node, logvar_node, mu_edge, logvar_edge
            mean_node, mean_edge = mean_node.data.detach(
            ).clone(), mean_edge.data.detach().clone()

            # concatenate all graphs in dataset
            if mean_node_cat is None:
                mean_node_cat = mean_node.cpu().numpy()
            else:
                mean_node_cat = np.concatenate(
                    [mean_node_cat, mean_node.cpu().numpy()], axis=0)
            if mean_edge_cat is None:
                mean_edge_cat = mean_edge.cpu().numpy()
            else:
                mean_edge_cat = np.concatenate(
                    [mean_edge_cat, mean_edge.cpu().numpy()], axis=0)

            if plot:
                if z_node_cat is None:
                    z_node_cat = z_node.cpu().detach().numpy()
                else:
                    z_node_cat = np.concatenate(
                        [z_node_cat, z_node.cpu().detach().numpy()], axis=0)

                if category_vec_cat is None:
                    category_vec_cat = category_vec.cpu().detach().numpy()
                else:
                    category_vec_cat = np.concatenate(
                        [category_vec_cat, category_vec.cpu().detach().numpy()], axis=0)

        if plot:
            self.plot_latent(z_node_cat, category_vec_cat, selected_objs=True)

        # center to zero
        mean_node_est = np.mean(mean_node_cat, axis=0,
                                keepdims=True)  # size 1*embed_dim
        mean_edge_est = np.mean(mean_edge_cat, axis=0,
                                keepdims=True)  # size 1*embed_dim
        mean_node_cat = mean_node_cat - mean_node_est
        mean_edge_cat = mean_edge_cat - mean_edge_est

        # final dim of mean_node_cat : (# of nodes in dataset, hidden_dim)
        n_node, d_node = mean_node_cat.shape[0], mean_node_cat.shape[1]
        # final dim of mean_edge_cat : (# of edges in dataset, hidden_dim)
        n_edge, d_edge = mean_edge_cat.shape[0], mean_edge_cat.shape[1]
        # dim of cov_node_est: (hidden_dim, hidden_dim)
        cov_node_est = np.zeros((d_node, d_node))
        cov_edge_est = np.zeros((d_edge, d_edge))
        for i in range(n_node):
            x = mean_node_cat[i]
            cov_node_est += 1.0 / (n_node - 1.0) * np.outer(x, x)
        # dim of mean_node_est: (hidden_dim, 1)
        mean_node_est = mean_node_est[0]

        for j in range(n_edge):
            x = mean_edge_cat[j]
            cov_edge_est += 1.0 / (n_edge - 1.0) * np.outer(x, x)
        mean_edge_est = mean_edge_est[0]

        return mean_node_est, cov_node_est, mean_edge_est, cov_edge_est

    def plot_latent(self, z_node, category_vec, selected_objs=True, long_list_objs=False):
        if selected_objs:
            z_node_selected = []
            category_selected = []
            obj_dic = {}
            if long_list_objs:
                selected_classes = ['ball', 'basket', 'bench', 'bed', 'box', 'cabinet', 'chair', 'armchair',
                                    'desk', 'door', 'floor', 'picture', 'sofa', 'couch', 'commode', 'monitor',
                                    'stool', 'tv', 'table']
            else:
                selected_classes = ['bed', 'chair', 'armchair', 'desk', 'door', 'floor', 'picture', 'sofa', 'couch',
                                    'stool', 'table']

            vocabulary = get_vocabulary(os.path.join(THREED_SSG_PLUS, "vocab", "objects.tsv"))
            
            # get latent vectors and category labels for selected objects
            for z, category in zip(z_node, category_vec):
                if category in list(obj_dic.keys()) and obj_dic.get(category) in selected_classes:
                    z_node_selected.append(z)
                    category_selected.append(obj_dic.get(category))
            z_node_selected, category_selected = np.array(
                z_node_selected), np.array(category_selected)

        # PCA for 2D plot
        pca = PCA(n_components=2)
        sns.set(rc={'figure.figsize': (11, 8)})

        if selected_objs:
            components = pca.fit_transform(z_node_selected)
            sns.scatterplot(
                x=components[:, 0], y=components[:, 1], hue=category_selected, palette=sns.color_palette(
                    "Paired", len(set(category_selected))))
        else:
            components = pca.fit_transform(z_node)
            sns.scatterplot(
                x=components[:, 0], y=components[:, 1], hue=category_vec)

        fig_dir = os.path.join(self.str_root, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, "posterior_train.png"))

    def loss(self, scores, targets, loss_type):
        """ Compute single loss"""
        if loss_type == "MSE":
            loss = nn.MSELoss(reduction='mean')(scores, targets)
        elif loss_type == "BCEwithLL":
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        elif loss_type == "L1":
            loss = nn.L1Loss()(scores, targets)
        elif loss_type == "CrossEntr":
            loss = nn.CrossEntropyLoss()(scores, targets)
        return loss

    def multi_layer_loss(self, loss_list):
        """ Compute aggregate loss"""
        loss = self.nn_multi_loss_layer.get_loss(loss_list=loss_list)
        sigma = self.nn_multi_loss_layer.sigma
        return loss, sigma
