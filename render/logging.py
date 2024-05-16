# import libraries
import numpy as np
import os
from scipy.stats import expon
from render.three_d import visualize

# import custom methods
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from render.three_d import get_lineset, get_object_mesh, get_shapenet_labels
from render.utils import SceneInfoContainer, compute_box_3d

def is_relevant(ratio: float) -> bool:
    """ Manage the frequencies of logging based on the ratio between current epoch / max num of epochs

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    rv = expon()
    prob_random = np.random.randint(100000000) / 100000000
    prob_insert = rv.pdf(ratio)

    if prob_insert > prob_random:
        return True
    else:
        return False


def add_nodes_open3d(GraphScene: SceneInfoContainer, representation: str, writer: object, step: int, iter: int):
    """ Add objects in tensorboard

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    num_nodes = GraphScene.nodes
    shapenet_labels = get_shapenet_labels()

    for node in range(GraphScene.nodes):

        # name of the objects
        obj_name = GraphScene.labels[node]

        # draw the boxes
        targets_points = compute_box_3d(dim=GraphScene.dim_targets[node], location=GraphScene.loc_targets[node],
                                        angle=np.argmax(GraphScene.ori_targets[node]) * 90 / GraphScene.oridim)
        lines = get_lineset(points=targets_points, rgb=[
                            0, 1, 0])  # green is target
        writer.add_3d(os.path.join("target", str(node) + obj_name, "box"),
                      to_dict_batch([lines]),
                      step=step,
                      max_outputs=1,
                      label_to_names=None,
                      description=None)

        scores_points = compute_box_3d(dim=GraphScene.dim_scores[node], location=GraphScene.loc_scores[node],
                                       angle=np.argmax(GraphScene.ori_scores[node]) * 90 / GraphScene.oridim)
        lines = get_lineset(points=scores_points, rgb=[
                            0, 0, 1])  # blue is score

        writer.add_3d(os.path.join("score", str(node) + obj_name, "box"),
                      to_dict_batch([lines]),
                      step=step,
                      max_outputs=1,
                      label_to_names=None,
                      description=None)

        # draw the meshes
        if representation == "mesh" and obj_name in shapenet_labels:
            if obj_name == "floor":
                color = [1, 1, 1]
            else:
                shade50 = node / (num_nodes + 1)
                color = [shade50, shade50, shade50]

            assert all(i > 0 for i in GraphScene.dim_scores[node]), print('GraphScene.dim_scores[node]', GraphScene.dim_scores[node])
            # only add mesh for groundtruth scene once
            if iter == 0:
                t = get_object_mesh(label=obj_name,
                                    color=color,
                                    dimension=GraphScene.dim_targets[node],
                                    location=GraphScene.loc_targets[node],
                                    angle=np.argmax(GraphScene.ori_targets[node]) * 90 / GraphScene.oridim,
                                    direction=GraphScene.directions[node])  # grey is target

                name = os.path.join("target", str(node) + obj_name, "mesh")
                writer.add_3d(name, data=to_dict_batch([t]), step=step,
                            max_outputs=1,
                            label_to_names=None, description=None)
            
            assert all(i > 0 for i in GraphScene.dim_scores[node]), print('GraphScene.dim_scores[node]', GraphScene.dim_scores[node])
            s = get_object_mesh(label=obj_name,
                                color=color,
                                dimension=GraphScene.dim_scores[node],
                                location=GraphScene.loc_scores[node],
                                angle=np.argmax(GraphScene.ori_scores[node]) * 90 / GraphScene.oridim,
                                direction=GraphScene.directions[node])  # grey is target

            name = os.path.join("score", str(node) + obj_name, "mesh")
            writer.add_3d(name,
                            data=to_dict_batch([s]),
                            step=step,
                            max_outputs=1,
                            label_to_names=None, description=None)

