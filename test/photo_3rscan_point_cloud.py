import pickle
import os
import warnings
import open3d as o3d
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


def main():

    #  directories
    path_to_saved_mesh = os.path.join("input", "data", "threed_ssg", "mesh")
    os.makedirs(path_to_saved_mesh, exist_ok=True)

    filename = os.path.join(
        "input", "data", "threed_ssg", "threed_ssg_subset_train.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    for g in range(len(data)):

        scan_id = data[g][1]

        # load the mesh
        mesh = get_texturemesh(scan_id=scan_id)

        visualize(mesh_list=[mesh], scan_id=scan_id, filename=os.path.join(
            path_to_saved_mesh, "ground_truth_"+scan_id), save_img=True)


if __name__ == "__main__":
    from render.three_d import get_texturemesh, visualize
    from data.data.threed_ssg.constructor import Dataset3DSSG
    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))
    main()
