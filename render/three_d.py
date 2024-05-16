""" Utils to render bounding boxes in Open3D"""

# import libraries
import open3d as o3d
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

# import custom methods
from render.data.three_scan.pipeline import download_scan
from render.data.three_scan.align import align_mesh
from render.utils import align_vector_to_another
from config.paths import SHAPENETSEM

def get_lineset(points: list, rgb=None):
    """ Build a line set from list of points
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    if rgb is None:
        rgb = [1, 1, 1]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6],
             [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = len(lines) * [rgb]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def rotate_mesh_axisZ(mesh: o3d.geometry.TriangleMesh, angle: float, measure: str = "degree"):
    if measure == "degree":
        angle = math.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return mesh.rotate(R=R, center=mesh.get_center())


def rotate_mesh_axisY(mesh: o3d.geometry.TriangleMesh, angle: float):
    angle = math.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return mesh.rotate(R=R, center=mesh.get_center())


def rotate_mesh_axisX(mesh: o3d.geometry.TriangleMesh, angle: float):
    angle = math.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    return mesh.rotate(R=R, center=mesh.get_center())


def rotate_mesh_upright(mesh: o3d.geometry.TriangleMesh, upright: np.array):
    axis, angle = align_vector_to_another(
        np.array(upright), np.array([0, 0, 1]))
    if axis is not None:
        if not math.isnan(axis[0]):
            axis_a = axis * angle
            mesh.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis_a), center=mesh.get_center())
    return mesh


def rotate_mesh_front(mesh: o3d.geometry.TriangleMesh, front: np.array):
    axis, angle = align_vector_to_another(np.array(front), np.array([0, 1, 0]))
    if axis is not None:
        if not math.isnan(axis[0]):
            axis_a = axis * angle
            mesh.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis_a), center=mesh.get_center())
    return mesh


def scale_mesh(mesh: o3d.geometry.TriangleMesh, dimension: np.array):
    dim_mesh = get_mesh_3d_dimension(mesh=mesh)
    scale_factor = get_scale_factor(dim_real=dimension, dim_mesh=dim_mesh)
    mesh.scale(scale=min(scale_factor), center=mesh.get_center())

    return mesh


def correct_front_vector(upright: np.array, front: np.array):
    if (front == np.array([0, 0, 1])).all():
        front = [-x for x in upright]
        return front
    elif (front == np.array([0, 0, -1])).all():
        front = upright
        return front
    else:
        return front


def get_scale_factor(dim_real: np.array, dim_mesh: np.array):
    # delta = 1
    delta_list = []
    for k in range(3):
        # delta = min(dim_real[k] / dim_mesh[k], delta)
        delta = min(dim_real[k] / dim_mesh[k], 1)
        delta_list.append(delta)
    return delta_list


def get_scaled_dim_mesh(dim_mesh, scale_factor):
    scaled_dim_mesh = [0, 0, 0]
    delta = min(scale_factor)
    delta_idx = scale_factor.index(delta)
    # scale the smallest ratio edge
    try:
        scaled_dim_mesh[delta_idx] = dim_mesh[delta_idx] * delta
    except IndexError as e:
        print('dim_mesh', dim_mesh)
        print('scale_factor', scale_factor)
        print('delta_idx', delta_idx)
        raise e
    # scale the rest by ratios
    for i, x in enumerate(scaled_dim_mesh):
        if x == 0:
            ratio = dim_mesh[i]/min(dim_mesh)
            assert ratio >= 1
            scaled_dim_mesh[i] = scaled_dim_mesh[delta_idx] * ratio

    return scaled_dim_mesh


def get_mesh_3d_dimension(mesh: o3d.geometry.TriangleMesh):
    if mesh.dimension() != 3:
        raise ValueError
    bb = mesh.get_axis_aligned_bounding_box()
    points = np.asarray(bb.get_box_points())
    dim_mesh = points[0] - points[4]
    dim_mesh = [abs(x) for x in dim_mesh]
    return np.array(dim_mesh)


def get_mesh_centroid(mesh: o3d.geometry.TriangleMesh):
    if mesh.dimension() != 3:
        raise ValueError
    bb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bb.get_center())
    return center


def get_shapenet_labels(root: str = None):
    df = pd.read_csv(os.path.join(SHAPENETSEM, "metadata_3dssg_dim.csv"))
    label_list = pd.unique(df["label"].str.split(
        ",", expand=True).stack()).tolist()
    return label_list


def get_dataframe_of_candidates(label: str, root: str = None):
    df = pd.read_csv(os.path.join(SHAPENETSEM, "metadata_3dssg_dim.csv"))
    candidate_idx = []
    for index, row in df.iterrows():
        if label in str(row['label']).split(','):
            candidate_idx.append(index)
    final_df = df.iloc[candidate_idx]
    return final_df


def get_mesh_by_id(obj_id: str):
    return o3d.io.read_triangle_mesh((os.path.join(SHAPENETSEM, "models-OBJ", "models", obj_id + ".obj")), True)


def get_best_candidate(candidates: pd.DataFrame, dimension: np.array, label: str):

    baseline = float('inf')
    final_obj_id = None

    for i, row in candidates.iterrows():
        obj_id = row["fullId"][4:]

        # get saved dimensions of candidates
        dim_mesh = [x + 1e-15 if x ==
                    0 else x for x in [row['dim_x'], row['dim_y'], row['dim_z']]]

        # compute the scale
        if label != "chair":
            scale_factor = get_scale_factor(
                dim_real=dimension, dim_mesh=dim_mesh)
        else:
            scale_factor = [1, 1, 1]
        # get the dimension of the scaled mesh
        scaled_dim_mesh = get_scaled_dim_mesh(
            dim_mesh=dim_mesh, scale_factor=scale_factor)

        distance = round(np.linalg.norm(scaled_dim_mesh - dimension), 3)
        if abs(distance) < baseline:
            baseline = abs(distance)
            final_obj_id = obj_id

    return final_obj_id


def get_best_candidate_mesh(obj_id: str, label: str, candidates: pd.DataFrame, dimension: np.array, rotation: float):
    mesh = get_mesh_by_id(obj_id)

    # raise error for corrupted mesh
    vertices = np.asarray(mesh.vertices).shape[0]
    if vertices < 4:
        raise Exception('Corrupted Mesh')

    # translate to positive axis
    mesh.translate((0, 0, 0), relative=False)
    origin = [abs(x) for x in mesh.get_min_bound()]
    mesh.translate(origin, relative=True)

    row = candidates[candidates['fullId'] == 'wss.'+obj_id]
    upright = [int(row['up_x']), int(row['up_y']), int(row['up_z'])]
    front = [int(row['front_x']), int(row['front_y']), int(row['front_z'])]

    # rotate mesh upright to z
    mesh = rotate_mesh_upright(mesh=mesh, upright=upright)

    # rotating the mesh upright can sometimes change the direction of the front vector
    front = correct_front_vector(front=front, upright=upright)

    # scale to same order of magnitude
    if label != "chair":
        mesh = scale_mesh(mesh=mesh, dimension=dimension)

    # setting the front to 0,1,0 .
    if (front == np.array([0, -1, 0])).all():
        mesh = rotate_mesh_axisZ(mesh=mesh, angle=180)
        front = np.array([0, 1, 0])

    # setting the front to 0,1,0 .
    if (front == np.array([-1, 0, 0])).all():
        mesh = rotate_mesh_axisZ(mesh=mesh, angle=180)
        front = np.array([1, 0, 0])

    # mesh = rotate_mesh_front(mesh=mesh, front=front)

    # rotate along z axis
    mesh = rotate_mesh_axisZ(
        mesh=mesh, angle=rotation, measure="degree")

    # play with the symmetry
    '''
    mesh = rotate_mesh_axisZ(mesh=mesh, angle=np.deg2rad(
        90) * (direction - 1), measure="radians")
    mesh = rotate_mesh_axisZ(mesh=mesh, angle=90)
    if direction == 3 or direction == 4:
        mesh = rotate_mesh_axisZ(mesh=mesh, angle=180)'''

    return mesh


def get_object_mesh(label: str, color: list, dimension: np.array, location: np.array,
                    angle: float, direction: int):
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    if label == "sofa":
        label = "couch"

    # query exact category
    # if label == "wall" or label == "ceiling" or label == "floor":
        # mesh = o3d.geometry.TriangleMesh.create_box(
        #    width=dimension[0], height=dimension[1], depth=dimension[2])
        # mesh = rotate_mesh_axisZ(mesh=mesh, angle=angle)
    # else:
    df = get_dataframe_of_candidates(label)
    # print(dimension)
    final_obj_id = get_best_candidate(
        candidates=df, dimension=dimension, label=label)
    if label == "commode":
        print('commode id: ', final_obj_id)

    if label == "floor":
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=dimension[0], height=dimension[1], depth=0.05)
        mesh = rotate_mesh_axisZ(mesh=mesh, angle=angle)
    else:
        mesh = get_best_candidate_mesh(obj_id=final_obj_id, label=label, candidates=df,
                                       dimension=dimension, rotation=angle)

    if mesh is None or label == "wall" or label == "ceiling":
        return None

    # place it in the room
    mesh.translate(location, relative=False)
    center_ = location - get_mesh_centroid(mesh=mesh)
    mesh.translate(center_, relative=True)

    # style it
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh


def save_dim_to_csv(root: str = None):
    if root is None:
        root = os.path.join("render", "data", "shapenet_sem")
    df_raw = pd.read_csv(os.path.join(root, "metadata_3dssg.csv"))
    dim_x, dim_y, dim_z = [], [], []

    for index, row in df_raw.iterrows():
        obj_id = row["fullId"][4:]
        mesh = get_mesh_by_id(obj_id, root)
        get_mesh_3d_dimension(mesh)

        vertices = np.asarray(mesh.vertices).shape[0]
        if vertices < 4:
            continue

        upright = [int(row['up_x']), int(row['up_y']), int(row['up_z'])]
        front = [int(row['front_x']), int(row['front_y']), int(row['front_z'])]

        # rotate mesh upright to z
        mesh = rotate_mesh_upright(mesh=mesh, upright=upright)
        dim_mesh = get_mesh_3d_dimension(mesh=mesh)
        dim_x.append(dim_mesh[0])
        dim_y.append(dim_mesh[1])
        dim_z.append(dim_mesh[2])

    df_raw['dim_x'] = dim_x
    df_raw['dim_y'] = dim_y
    df_raw['dim_z'] = dim_z
    df_final = df_raw[df_raw['dim_x'].notnull()]

    df_final.to_csv(os.path.join(root, "metadata_3dssg_dim.csv"), index=False)


def get_texturemesh(scan_id: str):
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    with open("paths.json") as f:
        paths = json.load(f)

    root = os.path.join("render", "data", "three_scan", "scans")
    os.makedirs(root, exist_ok=True)
    list_of_downloaded_scans = next(os.walk(os.path.join(
        "render", "data", "three_scan", "scans", ".")))[1]
    if scan_id not in list_of_downloaded_scans:
        download_scan(scan_id)
        align_mesh(scan_id)

    path = os.path.join(root, scan_id, "labels.instances.align.annotated.ply")
    mesh = o3d.io.read_triangle_mesh(path, True)
    mesh.compute_vertex_normals()
    return mesh


def apply_icp(source, target):
    print("Apply point-to-plane ICP")
    target.estimate_normals()
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    rotation = reg_p2l.transformation[:3, :3]
    translation = reg_p2l.transformation[:3, 3:]
    return rotation, translation


def display_inlier_outlier(cloud: o3d.geometry.PointCloud, ind: list):
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """

    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def visualize(mesh_list: list, filename: str, scan_id: str, save_img=False):
    """
    Visualize the scene, and optionally screenshot the window viewer with three camera angles

    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d

    """
    import shutil

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in mesh_list:
        viewer.add_geometry(geometry)

    if not save_img:
        try:
            viewer.run()
            viewer.destroy_window()
        except AttributeError:
            print("open3d Visualization Error")
    else:
        # get scene screenshot from different angles
        o3d_screenshot_mat_1 = viewer.capture_screen_float_buffer(True)
        o3d_screenshot_mat_1 = (
            255.0 * np.asarray(o3d_screenshot_mat_1)).astype(np.uint8)

        ctr = viewer.get_view_control()
        ctr.rotate(10.0, -270)

        o3d_screenshot_mat_2 = viewer.capture_screen_float_buffer(True)
        o3d_screenshot_mat_2 = (
            255.0 * np.asarray(o3d_screenshot_mat_2)).astype(np.uint8)

        ctr = viewer.get_view_control()
        ctr.rotate(270.0, 0)

        o3d_screenshot_mat_3 = viewer.capture_screen_float_buffer(True)
        o3d_screenshot_mat_3 = (
            255.0 * np.asarray(o3d_screenshot_mat_3)).astype(np.uint8)

        # save the results
        plt.imsave(filename+'_1.png', o3d_screenshot_mat_1)
        plt.imsave(filename+'_2.png', o3d_screenshot_mat_2)
        plt.imsave(filename+'_3.png', o3d_screenshot_mat_3)

        viewer.destroy_window()


def visualize_O3DVisualizer(list_: list):
    o3d.visualization.draw(geometry=list_)


def text_3d(text, pos, direction=None, degree=0.0, density=2, font='docs/assets/open-sans/OpenSans-Light.ttf',
            font_size=100):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(
        img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def vector_magnitude(vec):
    """
    https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    magnitude = np.sqrt(np.sum(vec ** 2))
    return (magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def create_arrow(scale=10):
    """
    https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = scale / 10
    cylinder_radius = scale / 20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
                                                        cone_height=cone_height,
                                                        cylinder_radius=0.5,
                                                        cylinder_height=cylinder_height)
    return (mesh_frame)


def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return (mesh)


def mesh_to_cp(textured_mesh: o3d.geometry.MeshBase, number_of_points: int = 50000) -> o3d.geometry.MeshBase:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return textured_mesh.sample_points_uniformly(number_of_points=number_of_points)


def cp_remove_stat_outliers(pcl: o3d.geometry.PointCloud, nb_neighbors: int = 30,
                            std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return pcl.remove_statistical_outlier(nb_neighbors, std_ratio)


def cp_remove_rad_outliers(pcl: o3d.geometry.PointCloud, nb_points: int = 16,
                           radius: float = .05) -> o3d.geometry.PointCloud:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return pcl.remove_radius_outlier(nb_points, radius)


def cp_points(pcl: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Author: Pietro Bonazzi
    https://github.com/uzh-rpg/scene_graph_3d
    """
    return np.asarray(pcl.points)

# PIPELINE

# textured_mesh=get_texturemesh(scan_id="19eda6f4-55aa-29a0-8893-8eac3a4d8193")
# pcl = mesh_to_cp(textured_mesh) # create a point cloud
# cl, ind = cp_remove_outliers(pcl) #normalize
# xyz = cp_points(pcl)
# cp_visualize_2d(xyz)
# xy = img_points(xyz)
# plot_img_points(xy)
# save_display_img(xy, "img.jpg")
