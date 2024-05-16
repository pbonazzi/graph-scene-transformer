import pickle
import os
import warnings
import open3d as o3d
import sys
import inspect
import random
import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def main():

    #  directories
    path_to_saved_shapenet_images = os.path.join(
        "input", "data", "threed_ssg", "shapenet")
    path_to_shapenet = os.path.join("render", "data", "shapenet_sem")

    os.makedirs(path_to_saved_shapenet_images, exist_ok=True)

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
        "input", "data", "threed_ssg", "threed_ssg_subset_train.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    for g in range(len(data)):

        graph = data[g][0]
        scan_id = data[g][1]

        dimensions = graph.ndata["raw_dimension"]
        location = graph.ndata["raw_location"]
        orientation = graph.ndata["orientation"]
        direction = graph.ndata["direction"]
        labels = get_labels(dataset_name="threed_ssg",
                            labels_type="nodes", graph=graph)

        target_mesh_list = []

        for node in range(graph.number_of_nodes()):

            # name of the objects
            obj_name = labels[node]

            # draw the meshes
            if obj_name in selected_obj_to_render and obj_name in shapenet_labels:
                if obj_name == "floor":
                    color = [0.3, 0.3, 0.3]
                else:
                    color = dict_shapenet_selected_objects[obj_name]

                mesh = get_object_mesh(label=obj_name, color=color, path_to_shapenet=path_to_shapenet, dimension=dimensions[node].numpy(), location=location[node].numpy(),
                                       angle=orientation[node].item(), direction=direction[node].item())

                if mesh is not None:
                    target_mesh_list.append(mesh)

        if len(target_mesh_list) > 3:
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            for geometry in target_mesh_list:
                viewer.add_geometry(geometry)

            o3d_screenshot_mat_1 = viewer.capture_screen_float_buffer(True)
            o3d_screenshot_mat_1 = (
                255.0 * np.asarray(o3d_screenshot_mat_1)).astype(np.uint8)
            plt.imsave(os.path.join(path_to_saved_shapenet_images,
                       scan_id+'.png'), o3d_screenshot_mat_1)
            plt.close()
            viewer.destroy_window()


if __name__ == "__main__":
    from data.utils import get_labels
    from render.three_d import visualize, get_shapenet_labels, get_object_mesh
    from data.data.threed_ssg.constructor import Dataset3DSSG
    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))
    main()
