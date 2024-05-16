import pickle
import os
import warnings
import open3d as o3d
import sys
import inspect
import random

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def main():

    #  directories
    path_to_saved_scene_diagram = os.path.join(
        "input", "data", "threed_ssg", "scene_graph_diagram")
    path_to_shapenet = os.path.join("render", "data", "shapenet_sem")

    os.makedirs(path_to_saved_scene_diagram, exist_ok=True)

    # load shapenet labels
    shapenet_labels = get_shapenet_labels(path_to_shapenet)
    selected_obj_to_render = ['ball', 'basket', 'bench', 'bed', 'box', 'cabinet', 'chair', 'armchair',
                              'desk', 'door', 'floor', 'picture', 'sofa', 'couch', 'commode', 'monitor',
                              'stool', 'tv', 'table', 'plant', 'pot', 'pillow', 'blanket']

    # create fixed dictionary of colors
    dict_shapenet_selected_objects = {}
    for i in range(len(selected_obj_to_render)):
        r = random.uniform(0, 1)
        g = random.uniform(0, 1)
        b = random.uniform(0, 1)
        rgb = [r, g, b]
        dict_shapenet_selected_objects[selected_obj_to_render[i]] = rgb

    filename = os.path.join(
        "input", "data", "threed_ssg", "threed_ssg_subset_val.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    for g in range(len(data)):

        graph = data[g][0]
        scan_id = data[g][1]
        labels = get_labels(dataset_name="threed_ssg",
                            labels_type="nodes", graph=graph)
        idx_fp_nodes = []

        for node in range(graph.number_of_nodes()):

            # name of the objects
            obj_name = labels[node]

            # draw the meshes
            if obj_name not in selected_obj_to_render or obj_name not in shapenet_labels:
                idx_fp_nodes.append(node)

        graph.remove_nodes(idx_fp_nodes)
        if graph.number_of_nodes() > 3:
            get_scene_graph_diagram(
                graph=graph, dataset_name="threed_ssg", filename=os.path.join(path_to_saved_scene_diagram, scan_id+".png"))


if __name__ == "__main__":
    from data.utils import get_labels
    from render.three_d import get_shapenet_labels
    from render.two_d import get_scene_graph_diagram
    from data.data.threed_ssg.constructor import Dataset3DSSG
    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))
    main()
