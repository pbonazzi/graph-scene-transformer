import csv
import pickle
import os
import sys
import inspect
import warnings
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)


def load_scene(data, scene_id):
    graph = data[scene_id][0]
    scan_id = data[scene_id][1]
    location = graph.ndata["raw_location"]
    dimension = graph.ndata["raw_dimension"]
    orientation = graph.ndata["orientation"]
    direction = graph.ndata["direction"]
    id = graph.ndata["id"]
    labels = get_labels(graph, "threed_ssg_subset", "nodes")

    path = os.path.join("test", "annotations", "data", str(scene_id))
    os.makedirs(path, exist_ok=True)
    # writer = SummaryWriter(path)
    mesh = get_texturemesh(scan_id)
    boxes, label_pcd = [], []
    # writer.add_3d("mesh", to_dict_batch([mesh]), step=0)

    for i in range(graph.number_of_nodes()):
        points = compute_box_3d(dim=dimension[i], location=location[i], angle=orientation[i],
                                measure="degree", direction=direction[i])
        lines = get_lineset(points=points, rgb=[0, 0, 1])  # blue is score
        label_pcd.append([labels[i], location[i]])
        # writer.add_3d(str(id[i].item()) + labels[i], to_dict_batch([lines]), step=0)
        boxes.append(lines)

    visualize([mesh, *boxes], label_pcd)

    get_scene_graph_diagram(
        graph, dataset_name='threed_ssg_subset', filename=scan_id)


def main():
    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

    filename = os.path.join("input", "data", "threed_ssg", "threed_ssg.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    issues, max_num, scene_id = [], 0, 0

    for g in range(len(data)):

        graph = data[g][0]
        scan_id = data[g][1]
        graph_obj_labels = get_labels(graph, "threed_ssg", "nodes")
        graph_rel_labels = get_labels(graph, "threed_ssg", "edges")
        est = 0

        src_nodes = graph.edges()[0]
        dst_nodes = graph.edges()[1]
        feat = graph.edata["feat"]

        location = graph.ndata["raw_location"]
        dimension = graph.ndata["raw_dimension"]
        ids = graph.ndata["id"]
        orientation = graph.ndata["orientation"]

        for i in range(len(feat)):
            # targets
            src_name = graph_obj_labels[src_nodes[i]]
            src_location = location[src_nodes[i]]
            src_dimension = dimension[src_nodes[i]]

            dst_name = graph_obj_labels[dst_nodes[i]]
            dst_location = location[dst_nodes[i]]
            dst_dimension = dimension[dst_nodes[i]]

            vol_src_targets = src_dimension[0] * \
                src_dimension[1] * src_dimension[2]
            vol_dst_targets = dst_dimension[0] * \
                dst_dimension[1] * dst_dimension[2]

            if feat[i] == 2:
                # left
                if src_location[0] > dst_location[0]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " left to ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 3:
                # right
                if src_location[0] < dst_location[0]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " right to ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 4:
                # front
                if src_location[1] > dst_location[1]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " front ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 5:
                # behind
                if src_location[1] < dst_location[1]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " behind ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 8:
                # bigger than
                if vol_src_targets < vol_dst_targets:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " bigger than ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 9:
                # smaller than
                if vol_src_targets > vol_dst_targets:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " smaller than ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 10:
                # higher
                if src_location[2] < dst_location[2]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " higher than ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

            elif feat[i] == 11:
                # lower
                if src_location[2] > dst_location[2]:
                    issues.append(
                        [g, scan_id, ids[src_nodes[i].item()].item(), src_name, " lower than ", ids[dst_nodes[i].item()].item(), dst_name])
                    est += 1

    with open("test/annotations/conflicts.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "scan_id", "src_id",
                        "src_label", "edge", "dst_id", "dst_label"])
        for num in range(len(issues)):
            writer.writerow(issues[num])
            print(issues[num])


if __name__ == "__main__":
    from data.data.threed_ssg.constructor import Dataset3DSSG
    from render.two_d import get_scene_graph_diagram
    from data.utils import get_labels
    from render.three_d import get_texturemesh, get_lineset, visualize
    from render.utils import compute_box_3d

    warnings.filterwarnings("ignore")
    o3d.utility.set_verbosity_level((o3d.utility.VerbosityLevel(0)))

    filename = os.path.join("input", "data", "threed_ssg", "threed_ssg.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # main()
    load_scene(data, 1251)
