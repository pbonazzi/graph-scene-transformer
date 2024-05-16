import torch
import numpy as np


def src_dot_dst(src_field: str, dst_field: str, name: str):
    """ Take the dot product between the src and dst field.
    """

    def func(obj):
        dim = obj.data[src_field].shape
        score_dict = {}
        for i in range(dim[1]):
            Q = obj.data[src_field][:, i:i + 1, :].squeeze()
            K = obj.data[dst_field][:, i:i + 1, :].squeeze()
            score_dict[name + "_score_of_head_" +
                       str(i)] = torch.matmul(Q, K.T)
        return score_dict

    return func


def scaling(num_heads: int, scale_constant: float, name: str):
    """ Scales the score by a constant
    """

    def func(obj):
        score_dict = {}
        for i in range(num_heads):
            score_dict[name + "_score_of_head_" + str(i)] = \
                obj.data[name + "_score_of_head_" + str(i)] / scale_constant
        return score_dict

    return func


def masking(num_heads: int, keys: np.ndarray, name: str):
    """ Substitutes the input at the indexes keys with -inf .
    """

    def func(obj):
        score_dict = {}
        for i in range(num_heads):
            score_dict[name + "_score_of_head_" +
                       str(i)] = obj.data[name + "_score_of_head_" + str(i)]
            score_dict[name + "_score_of_head_" + str(i)][keys] = torch.full_like(
                obj.data[name + "_score_of_head_" + str(i)][keys], float('-inf'))
        return score_dict

    return func


def score_dot_edge(num_heads):
    """ Take the dot product between the score and edge
    """

    def func(nodes):
        score_dict = {}
        for i in range(num_heads):
            E = nodes.data["edges_score_of_head_" + str(i)]
            # print(nodes.data.keys())
            # print('E', E.shape)
            score_dict["nodes_score_of_head_" +
                       str(i)] = nodes.data["nodes_score_of_head_" + str(i)] * E
        return score_dict

    return func


def score_plus_edge(num_heads):
    """ Sum the score and edge
    """

    def func(nodes):
        score_dict = {}
        for i in range(num_heads):
            E = nodes.data["edges_score_of_head_" + str(i)]
            score_dict["nodes_score_of_head_" +
                       str(i)] = nodes.data["nodes_score_of_head_" + str(i)] + E
        return score_dict

    return func


def softmax(num_heads):
    """ Softmax
    """

    def func(nodes):
        score_dict = {}
        for i in range(num_heads):
            score_dict["nodes_score_of_head_" + str(i)] = torch.softmax(nodes.data["nodes_score_of_head_" + str(i)],
                                                                        dim=0)
        return score_dict

    return func


def apply_attention(num_heads):
    def func(nodes):
        solution = torch.full_like(nodes.data["V_h"], 0)
        concat_score = nodes.data["nodes_score_of_head_0"]
        for i in range(num_heads):
            V = nodes.data["V_h"][:, i:i + 1, :].squeeze()
            score = nodes.data["nodes_score_of_head_" + str(i)]
            solution[:, i, :] = torch.matmul(score, V)
            if i == 0:
                continue
            concat_score = torch.cat((concat_score, score), dim=1)
        return {"h_out": solution, "concat_score": concat_score}

    return func


def apply_attention_on_edges(num_heads):
    def func(edges):
        solution = torch.full_like(edges.src["V_h"], 0)
        concat_score = edges.data["nodes_score_of_head_0"]
        for i in range(num_heads):
            V = edges.src["V_h"][:, i:i + 1, :].squeeze()
            score = edges.data["nodes_score_of_head_" + str(i)]
            solution[:, i, :] = torch.matmul(score, V)
            if i == 0:
                continue
            concat_score = torch.cat((concat_score, score), dim=1)
        return {"weV": solution, "concat_score": concat_score}

    return func


def copy(src, dst):
    """ Copy the src field input to the dst field .
    """

    def func(nodes):
        return {dst: nodes.data[src]}

    return func


####################################################################################################


def src_dot_dst_on_edg(src_field: str, dst_field: str, name: str, n_of_heads: int):
    def func(edges):
        score_dict = {}
        for i in range(n_of_heads):
            K = edges.src[src_field][:, i:i + 1, :].squeeze()
            Q = edges.dst[dst_field][:, i:i + 1, :].squeeze()
            score_dict[name + "_score_of_head_" +
                       str(i)] = torch.matmul(Q, K.T)
        return score_dict

    return func


def copy_field(src_field, out_field):
    """ Copy a edge field to an node .
    """

    def func(edges):
        return {out_field: (edges.data[src_field])}

    return func


def src_edge_node(src_field, out_field):
    """ Copy a node field to an edge .
    """

    def func(edges):
        return {out_field: (edges.src[src_field])}

    return func


def dst_edge_node(dst_field, out_field):
    """ Copy a node field to an edge field.
    """

    def func(edges):
        return {out_field: (edges.dst[dst_field])}

    return func


###########################################################################################
# from : https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_edge_layer.py

def src_multiply_dst(src_field, dst_field, out_field):
    def func(edges):
        # print((edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True).shape) # return tensor with size [# of edges, heads, 1]
        # return tensor with size [# of edges, heads, 64]
        # print((edges.src[src_field] * edges.dst[dst_field]).shape)
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling_multiply(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


def src_mul_edge():
    def func(edges):
        # clamp for softmax numerical stability
        return {"tmpVh": torch.mul(edges.src["V_h"], edges.data["score"])}
    return func


def edge_sum_src():
    def func(nodes):
        # clamp for softmax numerical stability
        return {"wV": nodes.mailbox["tmpVh"]}
    return func

# https://docs.dgl.ai/en/0.6.x/tutorials/models/4_old_wines/7_transformer.html


def src_dot_dst_sum(src_field: str, dst_field: str, out_field: str):
    """ Take the dot product between the src and dst field.
    """
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func
