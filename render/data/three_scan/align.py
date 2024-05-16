import numpy as np
from plyfile import PlyData
import os
import json
from shutil import copyfile


def resave_ply(filename_in, filename_out, matrix):
    """ Reads PLY file from filename_in, transforms with matrix and saves a PLY file to filename_out.
    """
    with open(filename_in, 'rb') as file:
        plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']
                      ['y'], plydata['vertex']['z'])).transpose()
    # shape = np.array(points.shape)
    points4f = np.insert(points, 3, values=1, axis=1)
    points = points4f * matrix
    plydata['vertex']['x'] = np.asarray(points[:, 0]).flatten()
    plydata['vertex']['y'] = np.asarray(points[:, 1]).flatten()
    plydata['vertex']['z'] = np.asarray(points[:, 2]).flatten()
    # make sure its binary little endian
    plydata.text = False
    plydata.byte_order = '<'
    # save
    plydata.write(filename_out)


def read_transform_matrix(filename):
    rescan2ref = {}
    with open(filename, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(
                        scans["transform"]).reshape(4, 4)
    return rescan2ref


def align_mesh(scan_id: str):
    root = os.path.join("render", "data", "three_scan")
    rescan2ref = read_transform_matrix(
        os.path.join(root, "metadata", "3RScan.json"))
    counter = 0
    if os.path.exists(os.path.join(root, "metadata", "rescans.txt")):
        with open(os.path.join(root, "metadata", "rescans.txt"), 'r') as f:
            for line in f:
                line_id = line.rstrip()
                if line_id != scan_id:
                    continue

                file_in = os.path.join(
                    root, "scans", scan_id, "labels.instances.annotated.v2.ply")
                file_out = os.path.join(
                    root, "scans", scan_id, "labels.instances.align.annotated.ply")
                if scan_id in rescan2ref:  # if not we have a hidden test scan
                    resave_ply(file_in, file_out, rescan2ref[scan_id])
                    return
                counter += 1
    counter = 0
    if os.path.exists(os.path.join(root, "metadata", "references.txt")):
        with open(os.path.join(root, "metadata", "references.txt"), 'r') as f:
            for line in f:
                line_id = line.rstrip()
                if line_id != scan_id:
                    continue
                file_in = os.path.join(
                    root, "scans", scan_id, "labels.instances.annotated.v2.ply")
                file_out = os.path.join(
                    root, "scans", scan_id, "labels.instances.align.annotated.ply")
                copyfile(file_in, file_out)
                counter += 1
