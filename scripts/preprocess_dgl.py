import os
import time
import csv
import torch
import dgl
import pickle
from tqdm import tqdm
import json,math, argparse
from dgl.data import DGLDataset
import pdb

def get_vocabulary(path: str):
    """ Read vocabulary of categories
    """

    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv)  # skip first row
    vocabulary = {}
    for rows in read_tsv:
        vocabulary[int(rows[0])] = rows[1]
    return vocabulary


def get_mean_and_std(graph, feature):

    assert feature in ["raw_location", "raw_dimension"]

    f = [g.ndata[feature] for g in graph]
    f = torch.cat(f, dim=0)

    f_x_mean, f_x_std = torch.mean(
        f[:, :1]), torch.std(f[:, :1])
    f_y_mean, f_y_std = torch.mean(
        f[:, 1:2]), torch.std(f[:, 1:2])
    f_z_mean, f_z_std = torch.mean(
        f[:, 2:3]), torch.std(f[:, 2:3])

    f_mean = [f_x_mean.item(
    ), f_y_mean.item(), f_z_mean.item()]

    f_std = [f_x_std.item(
    ), f_y_std.item(), f_z_std.item()]

    return f_mean, f_std

def euclidian_distance(vec1, vec2):
    return  torch.sqrt((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2 +(vec1[2]-vec2[2])**2)

class Scene3DSSG(DGLDataset):
    def __init__(self, bounding_boxes, triplets, objects, freq_classes, path):
        """ Class representing single scenes

            Parameters
            ----------
            - json data_bb (bounding_boxes.json), scene_graph (relationships.json), obj_graph (objects.json)
            - torch.Tensor embeds
            - int num_objects_type, max_objects_room

        Args:
            bounding_boxes (object)
        """
        self.bounding_boxes = bounding_boxes
        self.triplets = triplets
        self.objects = objects

        self.g = dgl.DGLGraph()
        # self.nodes_ids = list(self.bounding_boxes.keys())
        self.nodes_ids = list(self.objects.keys())
        
        if path == THREED_SSG_PLUS: 
            self.obj_vocabulary = get_vocabulary(
                os.path.join(THREED_SSG_PLUS, "vocab", "objects.tsv"))
            self.edge_vocabulary = get_vocabulary(
                os.path.join(THREED_SSG_PLUS, "vocab", "relationships.tsv"))
            floor_id=188
        else : 
            self.obj_vocabulary = get_vocabulary(
                os.path.join(OUTPUT_DIR, "3dfront", "processed_data", "vocab", "objects.tsv"))
            self.edge_vocabulary = get_vocabulary(
                os.path.join(OUTPUT_DIR, "3dfront", "processed_data", "vocab", "relationships.tsv"))
            floor_id=0
            
            
        self._add_nodes()
        if self.g.number_of_nodes() < 1:
            return 
        self.progressive_dict_of_classes = self._add_edges(freq_classes)
        self._normalize_according_to_floor_object(floor_id)

    def _add_nodes(self):
        """ Initialize the graph nodes, extract the labels
        """

        ori, dim, loc = [], [], []

        # collect values for normalization
        xmin, xmax, ymin, ymax, zmin, zmax = float("inf"), float("-inf"), float("inf"), float("-inf"), float(
            "inf"), float("-inf")
        lmin, lmax, wmin, wmax, hmin, hmax = float("inf"), float("-inf"), float("inf"), float("-inf"), float(
            "inf"), float("-inf")

        for node_id in self.nodes_ids:
            label = self.objects.get(node_id)
            global_id = [k for k, v in self.obj_vocabulary.items() if v == label][0]
            if node_id not in self.bounding_boxes.keys():
                continue
            if "direction" not in self.bounding_boxes[node_id].keys():
                direction = 1
            else:
                direction = self.bounding_boxes[node_id]["direction"]

            # https://github.com/he-dhamo/graphto3d/blob/main/dataset/dataset.py
            angle = self.bounding_boxes[node_id]['param7'][6] % 90

            # collect values for normalization
            centroid = self.bounding_boxes[node_id]['param7'][3:-1]
            dimension = self.bounding_boxes[node_id]['param7'][0:3]
            if direction == 2 or direction == 4:
                temp = dimension[0]
                dimension[0] = dimension[1]
                dimension[1] = temp
            [x, y, z] = centroid
            [l, w, h] = dimension

            l = abs(l) 
            w = abs(w) 
            h = abs(h)
                
            if x > xmax:
                xmax = x
            elif x < xmin:
                xmin = x

            if y > ymax:
                ymax = y
            elif y < ymin:
                ymin = y

            if z > zmax:
                zmax = z
            elif z < zmin:
                zmin = z

            if l > lmax:
                lmax = l
            elif l < lmin:
                lmin = l

            if w > wmax:
                wmax = w
            elif w < wmin:
                wmin = w

            if h > hmax:
                hmax = h
            elif h < hmin:
                hmin = h

            self.g = dgl.add_nodes(self.g, 1, {'id': torch.tensor([int(node_id)]),
                                               "global_id": torch.tensor([global_id]),
                                               "direction": torch.tensor([direction]),
                                               "orientation": torch.tensor([int(angle)])})

            ori.append(angle)  # orientation # ori = [ang,..]
            dim.append(dimension)  # dimension # dim = [[l,w,h],..]
            loc.append(centroid)  # location # loc = [[x,y,z],..]

        self.loc, self.dim = loc, dim
        self.dim_range = [lmin, lmax, wmin, wmax, hmin, hmax]
        self.loc_range = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.max_distance = euclidian_distance(torch.tensor([xmin, ymin, zmin]), torch.tensor([xmax, ymax, zmax]))

        self.g.ndata['raw_location'] = torch.tensor(loc)
        self.g.ndata['raw_dimension'] = torch.tensor(dim)
        self.g.ndata["maximum_distance"] = torch.full_like(input=torch.tensor(dim), fill_value= self.max_distance)

    def _add_edges(self, freq_classes):
        """ create edges
        """
        if self.g.number_of_nodes() < 1:
            return freq_classes

        nodes_id = self.g.ndata['id'].tolist()
        edge_list, edge_features = [], []

        for i in range(len(self.triplets)):

            try:

                src = nodes_id.index(int(self.triplets[i][0]))
                dst = nodes_id.index(int(self.triplets[i][1]))
                edge_list.append([src, dst])
                edge_features.append(int(self.triplets[i][2]))
                freq_classes[self.triplets[i][2]] += 1
            except ValueError:
                '''
                Some relationships triplets contain nodes that don't have a 
                bounding box. We ignore those triplets that raise value errors. 
                '''
                continue

        edge_list = torch.tensor(edge_list)
        edge_features = torch.tensor(edge_features)

        for src, dst in edge_list:
            self.g.add_edges(src.item(), dst.item())

        self.g.edata['feat'] = edge_features
        
        for i in range(self.g.number_of_nodes()):
            for j in range(self.g.number_of_nodes()-1):
                if euclidian_distance(self.g.ndata["raw_location"][i],self.g.ndata["raw_location"][j]) < self.max_distance/2:
                    freq_classes[27] += 2
        
        return freq_classes

    def _normalize_according_to_floor_object(self, floor_id):
        """ normalize location and orientation of all objects according to position of the floor plan
        """

        idx_floor = (self.g.ndata['global_id'] == floor_id).nonzero(
            as_tuple=True)[0]

        if len(idx_floor) == 1:
            self.g.ndata['raw_location'] = self.g.ndata['raw_location'] - \
                self.g.ndata['raw_location'][idx_floor]
            self.g.ndata['orientation'] = self.g.ndata['orientation'] - \
                self.g.ndata['orientation'][idx_floor]
        elif len(idx_floor) > 1:
            origin_location = 0
            origin_orientation = 0
            for i in range(len(idx_floor)):
                origin_location += self.g.ndata['raw_location'][idx_floor[i]]
                origin_orientation += self.g.ndata['orientation'][idx_floor[i]]

            self.g.ndata['raw_location'] = self.g.ndata['raw_location'] - \
                origin_location/len(idx_floor)
            self.g.ndata['orientation'] = self.g.ndata['orientation'] - \
                int(origin_orientation/len(idx_floor))

        for i in range(self.g.number_of_nodes()):
            while self.g.ndata['orientation'][i] not in range(0, 90):
                if self.g.ndata['orientation'][i] > 90:
                    self.g.ndata['orientation'][i] -= 90
                else:
                    self.g.ndata['orientation'][i] += 90

    def get_triple_local_ids(self):
        """
        return list of triplets in [source node object local id, destination node object local id, edge type id] format
        """
        triplet = [x[:-1] for x in self.triplets]
        triplets_in_ids = []

        for src, dst, feat in zip(self.g.edges()[0], self.g.edges()[1], self.g.edata['feat']):
            src_local_id, dst_local_id = self.nodes_ids[int(
                src)], self.nodes_ids[int(dst)]
            tri_from_g = [int(src_local_id), int(dst_local_id), int(feat)]
            assert tri_from_g in triplet, f"triplet from graph does not match raw data: {tri_from_g}"
            triplets_in_ids.append(tri_from_g)

        return triplets_in_ids


class Dataset3DSSG(DGLDataset):
    def __init__(self, split: str, path: str):
        """ Class representing all dataset
        """

        # meta data
        if path==THREED_SSG_PLUS:
            self.n_of_relationships_type = 26
            self.n_of_objects = 529
        else:
            self.n_of_relationships_type = 10
            self.n_of_objects = 86
            
        # paths
        if split == 'train':
            relationships_path = os.path.join(path, "relationships_train.json")
        elif split == 'val':
            relationships_path = os.path.join(path, "relationships_validation.json")

        bounding_boxes_path = os.path.join(path, "objects.json")
        self.bounding_boxes = json.load(open(bounding_boxes_path, ))

        relationships = json.load(open(relationships_path, ))
        self.graphs = relationships['scans']
        self.n_of_graphs = len(self.graphs)
        
        if path==THREED_SSG_PLUS:
            self.scan_ids = [g['scan'] + '_' + str(g['split']) for g in self.graphs]
        else:
            self.scan_ids = [g['scan'] for g in self.graphs]

        self.graph_lists = []

        self._prepare(path)

    def _prepare(self, path):
        loc_range, dim_range = [], []
        
        freq_classes = {}
        for i in range(28):
            freq_classes[i] = 0

        for i in tqdm(range(self.n_of_graphs), desc="Scanning the graphs"):
            """ For each scan in the dataset transform it in dgl format"""

            triplets = self.graphs[i]['relationships']

            scene_graph_id = self.graphs[i]['scan']
            objects = self.graphs[i]['objects']
            bounding_boxes = self.bounding_boxes[scene_graph_id]
            scene = Scene3DSSG(bounding_boxes, triplets, objects, freq_classes, path)
            if scene.g.number_of_nodes() < 1:
                continue
            freq_classes = scene.progressive_dict_of_classes

            self.graph_lists.append(scene.g)
            dim_range.append(scene.dim_range)
            loc_range.append(scene.loc_range)

        # location
        # get range of location in the dataset for normalization
        xmin = min(loc_range, key=lambda x: x[0])[0]
        xmax = max(loc_range, key=lambda x: x[1])[1]
        ymin = min(loc_range, key=lambda x: x[2])[2]
        ymax = max(loc_range, key=lambda x: x[3])[3]
        zmin = min(loc_range, key=lambda x: x[4])[4]
        zmax = max(loc_range, key=lambda x: x[5])[5]
        self.denormalize_location = [xmin, xmax, ymin, ymax, zmin, zmax]

        # dimension
        # get range of dimension in the dataset for normalization
        lmin = min(dim_range, key=lambda x: x[0])[0]
        lmax = max(dim_range, key=lambda x: x[1])[1]
        wmin = min(dim_range, key=lambda x: x[2])[2]
        wmax = max(dim_range, key=lambda x: x[3])[3]
        hmin = min(dim_range, key=lambda x: x[4])[4]
        hmax = max(dim_range, key=lambda x: x[5])[5]
        self.denormalize_dimension = [lmin, lmax, wmin, wmax, hmin, hmax]
        
        # classes 
        self.classes = freq_classes

    def _normalize_data(self):
        self.graph_lists = transform_graphs(
            func=normalize_graph, data=self.graph_lists, arg=self.N)

    def add_laplacian_positional_encodings(self, pos_enc_dim):
        self.graph_lists = transform_graphs(
            func=laplacian_positional_encoding, data=self.graph_lists, arg=pos_enc_dim)

    def add_wl_positional_encodings(self):
        self.graph_lists = transform_graphs(
            func=wl_positional_encoding, data=self.graph_lists)

    def __len__(self):
        return len(self.graph_lists)

    def __getitem__(self, idx: int):
        return self.graph_lists[idx], self.scan_ids[idx]


def main():
    
    # parser
    parser = argparse.ArgumentParser(
        description='I solemnly swear that I am up to no good.')

    parser.add_argument('--dataset', '--d', required=True,
                        help='dataset name')
    args = parser.parse_args()
    
    
    t0 = time.time()
    
    if args.dataset == "3dssg":
        path = THREED_SSG_PLUS
    elif args.dataset == "3dfront":
        path =  os.path.join(OUTPUT_DIR, args.dataset, "processed_data")
    
    train_data = Dataset3DSSG(split='train', path=path)

    # z-score normalization
    dim_mean, dim_std = get_mean_and_std(train_data.graph_lists, "raw_dimension")
    loc_mean, loc_std = get_mean_and_std(train_data.graph_lists, "raw_location")

    dict_stats = {}
    dict_stats["train_location_mean"] = loc_mean
    dict_stats["train_location_std"] = loc_std
    dict_stats["train_dimension_mean"] = dim_mean
    dict_stats["train_dimension_std"] = dim_std

    train_data.N = Normalizer(dict_stats=dict_stats)
    train_data._normalize_data()
    train_data.add_wl_positional_encodings()
    train_data.add_laplacian_positional_encodings(pos_enc_dim=4)

    root = os.path.join(OUTPUT_DIR, args.dataset, "processed_data")
    os.makedirs(root, exist_ok=True)
    
    with open(os.path.join(root, args.dataset + "_train.pkl"), 'wb') as f:
        pickle.dump(train_data, f)
    
    val_data = Dataset3DSSG(split='val', path=path)
    val_data.N = Normalizer(dict_stats=dict_stats)
    val_data._normalize_data()
    val_data.add_wl_positional_encodings()
    val_data.add_laplacian_positional_encodings(pos_enc_dim=4)

    with open(os.path.join(root, args.dataset + "_val.pkl"), 'wb') as f:
        pickle.dump(val_data, f)

    print("Time taken: {:.4f}s".format(time.time() - t0))


if __name__ == "__main__":
    import inspect
    import sys
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    from data.positional_encoding import laplacian_positional_encoding, wl_positional_encoding
    from data.utils import make_full_graph, transform_graphs, normalize_graph
    from data.normalization import Normalizer
    from config.paths import  OUTPUT_DIR, THREED_SSG_PLUS


    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1

    main()
