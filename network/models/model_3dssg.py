"""
    Utility file to select model as selected by the user
"""
# import libraries
import torch
import torch.nn as nn
import dgl

from network.layers.multi_layer_perceptron import build_mlp
from data.utils import get_dimension, get_orientation, get_location
from scripts.preprocess_dgl import euclidian_distance

class Model3DSSG(nn.Module):

    def __init__(self, params: dict, device: torch.device):
        super().__init__()
        self.device = device

        # bootstrapping
        embed_dim = int(params["bbox_embed"]/2)
        self.input_orientation_embed = nn.Embedding(24, embed_dim)
        self.input_size_embed = nn.Linear(3, embed_dim)

        self.input_subject_embed = nn.Linear(
            embed_dim*2, params["hidden_dim"])
        self.input_object_embed = nn.Linear(
            embed_dim*2, params["hidden_dim"])
        self.input_translation_embed = nn.Linear(3, params["hidden_dim"])

        self.mlp = build_mlp(
            dim_list=[params["hidden_dim"]*3, params["hidden_dim"]*2, params["hidden_dim"], params["number_of_edge_classes"]+1])

        self.num_of_classes = params["number_of_edge_classes"]        

    def forward(self, graph: dgl.DGLGraph):
        # bootstrapping
        graph = graph.to(self.device).local_var()
        num_nodes = graph.number_of_nodes()

        orientation_vec = get_orientation(
            graph=graph, device=self.device, ohc=False)
        orientation_embed = self.input_orientation_embed(orientation_vec).to(self.device)

        dimension = get_dimension(
            graph=graph, decomposed=False, normalized=True).to(self.device)
        dimension_embed = self.input_size_embed(dimension).to(self.device)

        # targets
        targets = []
        scores = []
        features = graph.edata["feat"]

        location = get_location(graph=graph, decomposed=False).to(self.device)
        raw=graph.ndata["raw_location"]
        distance = torch.zeros((num_nodes, num_nodes)).to(self.device)

        for i in range(num_nodes):

            idx_of_edg_types = graph.out_edges(i, form="all")[2]
            real_output_nodes = graph.out_edges(i)[1]

            # initialize a teacher supervision with a dictionary of node relationships 
            node_i_target_edges = {
                node: [self.num_of_classes] for node in range(num_nodes)}
            for num, idx in enumerate(idx_of_edg_types):
                if node_i_target_edges[real_output_nodes[num].item()] == [self.num_of_classes]:
                    node_i_target_edges[real_output_nodes[num].item()] = [features[idx].item()]
                else:
                    node_i_target_edges[real_output_nodes[num].item()].append(
                        features[idx].item())


            for j in range(num_nodes):
                # distance check
                if distance[i][j] == distance[j][i]:
                    distance[i][j] = euclidian_distance(raw[i], raw[j])

                if distance[i][j] < graph.ndata["maximum_distance"][0][0]/2:                    
                    relative_location = self.input_translation_embed(location[i]-location[j]).to(self.device)

                    # teacher supervision for number of edges
                    for pred in range(len(node_i_target_edges[j])):

                        subject_embed = self.input_subject_embed(torch.cat(
                            [dimension_embed[i], orientation_embed[i]])).to(self.device)
                        object_embed = self.input_object_embed(torch.cat(
                            [dimension_embed[j], orientation_embed[j]])).to(self.device)

                        edg_input = torch.cat(
                            [subject_embed, relative_location, object_embed], dim=-1).to(self.device)

                        output = self.mlp(edg_input)

                        scores.append(output)
                        targets.append(node_i_target_edges[j][pred])

        scores = torch.stack(scores).to(self.device)
        targets = torch.tensor(targets).to(self.device)
        return scores, targets

    def inference(self, graph_without_edges: dgl.DGLGraph):

        # bootstrapping
        graph = graph_without_edges.to(self.device).local_var()
        num_nodes = graph.number_of_nodes()

        orientation_vec = get_orientation(
            graph=graph, device=self.device, ohc=False)
        orientation_embed = self.input_orientation_embed(orientation_vec)

        dimension = get_dimension(
            graph=graph, decomposed=False, normalized=True)

        dimension_embed = self.input_size_embed(dimension)

        # targets

        location = get_location(graph=graph, decomposed=False)
        distance = torch.zeros((num_nodes, num_nodes)).to(self.device)
        
        feat = []

        for i in range(num_nodes):

            for j in range(num_nodes):

                # distance check
                if distance[i][j] == distance[j][i]:
                    distance[i][j] = torch.sqrt((location[i][0]-location[j][0])**2+(
                        location[i][1]-location[j][1])**2+(location[i][2]-location[j][2])**2)
                thresh_com = distance[i][j]
                if thresh_com < self.distance_threshold:
                    relative_location = self.input_translation_embed(
                        location[i]-location[j])

                    edge_pair_generation = True

                    # generation until end token
                    checker = 0
                    while True:

                        subject_embed = self.input_subject_embed(torch.cat(
                            [dimension_embed[i], orientation_embed[i]])).to(self.device)
                        object_embed = self.input_object_embed(torch.cat(
                            [dimension_embed[j], orientation_embed[j]])).to(self.device)

                        edg_input = torch.cat(
                            [subject_embed, relative_location, object_embed], dim=-1).to(self.device)

                        output = torch.argmax(self.mlp(edg_input))

                        feat.append(output)
                        graph.add_edges(i, j)

                        if output == self.num_of_classes or checker>10:
                            break
                        checker += 1
                        
        graph.edata["feat"] = torch.tensor(feat)
        return graph

    def loss(self, scores, targets, loss_type, weight=None):
        """ Compute single loss"""
        if loss_type == "MSE":
            loss = nn.MSELoss(reduction='mean')(scores, targets)
        elif loss_type == "BCEwithLL":
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        elif loss_type == "L1":
            loss = nn.L1Loss()(scores, targets)
        elif loss_type == "CrossEntr":
            loss = nn.CrossEntropyLoss(weight=weight)(scores, targets)
        return loss
