import open3d as o3d
import numpy as np
from tqdm import tqdm
import json
import os, math
from collections import defaultdict
import matplotlib.pyplot as plt

def quaternion_to_matrix(args):
    tx = args[0] + args[0]
    ty = args[1] + args[1]
    tz = args[2] + args[2]
    twx = tx * args[3]
    twy = ty * args[3]
    twz = tz * args[3]
    txx = tx * args[0]
    txy = ty * args[0]
    txz = tz * args[0]
    tyy = ty * args[1]
    tyz = tz * args[1]
    tzz = tz * args[2]

    result = np.zeros((3, 3))
    result[0, 0] = 1.0 - (tyy + tzz)
    result[0, 1] = txy - twz
    result[0, 2] = txz + twy
    result[1, 0] = txy + twz
    result[1, 1] = 1.0 - (txx + tzz)
    result[1, 2] = tyz - twx
    result[2, 0] = txz - twy
    result[2, 1] = tyz + twx
    result[2, 2] = 1.0 - (txx + tyy)
    return result


def quaternion_to_z_angle(rotation):
    ref = [0, 0, 1]
    # axis = np.cross(ref, rotation[1:])
    theta = np.arccos(np.dot(ref, rotation[1:]))*2
    theta = 180*theta/(math.pi)

    return theta

def get_lineset(points: list, rgb=None):
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




def main():
    
    parser = argparse.ArgumentParser(
    description='I solemnly swear that I am up to no good.')

    parser.add_argument('--visualize', '--v', default=False,
                        help="visualize scenes")
    parser.add_argument('--floorplan', '--f', default=False,
                        help="create floor plans")
    args = parser.parse_args()
    
    
    # Parse the model info
    model_dict = {}
    model_info = json.load(open(os.path.join(THREED_FUTURE, "model_info.json"),))
    for i in range(len(model_info)):
        model_dict[model_info[i]["model_id"]] = {"category": model_info[i]["category"],
                                                "super-category": model_info[i]["super-category"]}

    path_to_scene_layouts = [
        os.path.join(THREED_FRONT, f)
        for f in sorted(os.listdir(THREED_FRONT))
        if f.endswith(".json")]


    # Start parsing the dataset
    print("Loading dataset ", end="")
    count, total = 0, 0
    threed_front_data = {}
    objects_vocab = {}

    for i, m in tqdm(enumerate(path_to_scene_layouts)):
        
        # Load the layout file (threed_front)
        with open(m) as f:
            data = json.load(f)

            # Parse the furniture of the scene
            furniture_in_scene = defaultdict()
            for ff in data["furniture"]:
                if "valid" in ff and "size" in ff and ff["valid"]:
                    furniture_in_scene[ff["uid"]] = dict(
                        model_uid=ff["uid"],
                        model_jid=ff["jid"],
                        model_size=ff["size"]
                    )

            # Parse walls, doors, windows etc.
            meshes_in_scene = defaultdict()
            for mm in data["mesh"]:
                meshes_in_scene[mm["uid"]] = dict(
                    mesh_uid=mm["uid"],
                    mesh_jid=mm["jid"],
                    mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                    mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                    mesh_type=mm["type"]
                )

            # Parse the rooms of the scene
            scene = data["scene"]
            saved_scene = {}

            # Keep track of the parsed rooms
            for rr in scene["room"]:

                # new room
                room = o3d.geometry.TriangleMesh()
                bbox_list = []
                furnitures, geom = 0, 0  # visualize only scenes with furniture
                room_id = rr["instanceid"].lower()+"-"+data["uid"]
                candidate_room = {}

                for j, cc in enumerate(rr["children"]):

                    # skip small/big scenes
                    if any((si < 1e-5 or si > 5) for si in cc["scale"]):
                        break

                    if cc["ref"] in furniture_in_scene:
                        tf = furniture_in_scene[cc["ref"]]
                        furnitures += 1

                        # skip others
                        if model_dict[tf["model_jid"]]["category"] is None:
                            continue

                        # load
                        path_to_obj_in_future = os.path.join(
                            THREED_FUTURE, tf["model_jid"], "raw_model.obj")
                        mesh = o3d.io.read_triangle_mesh(
                            path_to_obj_in_future, True)
                        mesh.compute_vertex_normals()

                        # position mesh
                        rot = cc["rot"]
                        scale = cc["scale"]
                        pos = cc["pos"]
                        v = np.asarray(mesh.vertices) * scale
                        rotMatrix = quaternion_to_matrix(rot)
                        R = np.array(rotMatrix)
                        v = np.transpose(v)
                        v = np.matmul(R, v)
                        v = np.transpose(v)
                        v = v + pos
                        mesh.vertices = o3d.utility.Vector3dVector(v)

                        bbox = mesh.get_axis_aligned_bounding_box()
                        bbox_points = bbox.get_box_points()
                        bbox_list.append(bbox)
                        center, dimension = [], []
                        for i in range(3):
                            axis_max = np.max(np.asarray(bbox_points)[:, i:i+1])
                            axis_min = np.min(np.asarray(bbox_points)[:, i:i+1])
                            center.append((axis_max+axis_min)/2)
                            dimension.append(abs(axis_max-axis_min))

                        try:
                            mes_data = {
                                "location": center,
                                "dimension": dimension,
                                "param7": [*center, *dimension, quaternion_to_z_angle(rot)],
                                "rotation": quaternion_to_z_angle(rot),
                                "category": model_dict[tf["model_jid"]]["category"].lower(),
                                "super-category": model_dict[tf["model_jid"]]["super-category"].lower(),
                                "vis_id": tf["model_jid"],
                                "vis_scale": cc["scale"]
                            }
                            if not model_dict[tf["model_jid"]]["category"].lower() in objects_vocab.values():
                                objects_vocab[len(objects_vocab.values())] = model_dict[tf["model_jid"]]["category"].lower()
                        except:
                            breakpoint()

                        candidate_room[j] = mes_data

                        room += mesh

                    elif cc["ref"] in meshes_in_scene:
                        mf = meshes_in_scene[cc["ref"]]
                        geom += 1

                        # load
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(
                            mf["mesh_xyz"])
                        mesh.triangles = o3d.utility.Vector3iVector(
                            mf["mesh_faces"])
                        mesh.compute_vertex_normals()

                        # position mesh
                        rot = cc["rot"]
                        scale = cc["scale"]
                        pos = cc["pos"]
                        v = np.asarray(mesh.vertices) * scale
                        rotMatrix = quaternion_to_matrix(rot)
                        R = np.array(rotMatrix)
                        v = np.transpose(v)
                        v = np.matmul(R, v)
                        v = np.transpose(v)
                        v = v + pos
                        mesh.vertices = o3d.utility.Vector3dVector(v)

                        # data
                        bbox = mesh.get_axis_aligned_bounding_box()
                        bbox_points = bbox.get_box_points()
                        bbox_list.append(bbox)
                        center, dimension = [], []
                        for i in range(3):
                            axis_max = np.max(np.asarray(bbox_points)[:, i:i+1])
                            axis_min = np.min(np.asarray(bbox_points)[:, i:i+1])
                            center.append((axis_max+axis_min)/2)
                            dimension.append(abs(axis_max-axis_min))

                        try:
                            mes_data = {
                                "location": center,
                                "param7": [*center, *dimension, quaternion_to_z_angle(rot)],
                                "dimension": dimension,
                                "rotation": quaternion_to_z_angle(rot),
                                "category": mf["mesh_type"].lower(),
                                "super-category": None,
                                "vis_id": None,
                                "vis_scale": cc["scale"]
                            }
                            if not mf["mesh_type"].lower() in objects_vocab.values():
                                objects_vocab[len(objects_vocab.values())] = mf["mesh_type"].lower()
                        except:
                            breakpoint()
                            

                        candidate_room[j] = mes_data

                        room += mesh
                    else:

                        count += 1
                
                        continue

                
                if np.asarray(room.triangles).shape[0] == 0:
                    continue
                if len(candidate_room) < 3 :
                    continue
                
                
                total += len(rr["children"])

                if args.floorplan :
                    pcl = room.sample_points_uniformly(number_of_points=10000)
                    xy=np.asarray(pcl.points)
                    
                    plt.figure(figsize=(5.5, 5.5), dpi=80)  # 1 in 	96 pixel (X)
                    plt.plot(xy[:, 0], xy[:, 2], 's', color="black")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.axis('off')
                    dir_name = os.path.join(OUTPUT_DIR, "3dfront", "processed_data", "floorplans")
                    os.makedirs(dir_name, exist_ok=True)
                    plt.savefig(os.path.join(dir_name, room_id + ".jpg"))
                    plt.close()
                    
                else:
                    threed_front_data[room_id] = candidate_room

    if not args.floorplan :
        dir = os.path.join(OUTPUT_DIR, "3dfront", "processed_data")
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'objects.json'), 'w') as f:
            json.dump(threed_front_data, f)
        
    with open(os.path.join(OUTPUT_DIR, "3dfront", "processed_data", "vocab", 'objects.tsv'), 'w') as f:
        f.write("id, label\n")
        for i, key in enumerate(objects_vocab.keys()):
            f.write("%s, %s\n" % (key, objects_vocab[key]))
    
if __name__ == "__main__":
    import inspect
    import sys, argparse
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)
    from config.paths import THREED_FRONT, THREED_FUTURE, OUTPUT_DIR
    sys.setrecursionlimit(2097152)


    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1


    main()