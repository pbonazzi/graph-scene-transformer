import open3d as o3d
import numpy as np
from tqdm import tqdm
import json
import os, csv
from collections import defaultdict
import math, argparse

def euclidian_distance(vec1, vec2):
    return  math.sqrt((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2 +(vec1[2]-vec2[2])**2)

def volume(vec):
    return  vec[0]*vec[1]*vec[2]

def main():
    
    parser = argparse.ArgumentParser(
        description='I solemnly swear that I am up to no good.')
    parser.add_argument('--val', default=False,
                        help="Split")
    args = parser.parse_args()
    


    dir = os.path.join(OUTPUT_DIR, "3dfront", "processed_data")
    os.makedirs(dir, exist_ok=True)   
         
         
         
    with open(os.path.join(dir, 'objects.json'),) as f:
        data = json.load(f)
        
    
    relationship_json = {"scans": []}
    n_of_rooms = len(data.keys())
    for iter, key in tqdm(enumerate(data.keys())):
        
        if (not args.val) and iter in range(0, int(n_of_rooms*0.2)):
            continue
        
        elif (args.val) and (iter in range(int(n_of_rooms*0.2), n_of_rooms)):
            continue

        room = {}
        room["scan"] = key
        room["objects"] = {} 
        objects_info = data[key]
        number_of_objects = len(objects_info)
        
        min_d, max_d = [float("inf")]*3, [float("-inf")]*3
        for obj in objects_info:
            # if obj["category"] == "baseboard":
            #     continue
            room["objects"][obj["id_sg"]] = obj["category"]
            loc =obj["location"]
            min_d = [min(min_d[0], loc[0]), min(min_d[1], loc[1]), min(min_d[2], loc[2])]
            max_d = [max(max_d[0], loc[0]), max(max_d[1], loc[1]), max(max_d[2], loc[2])]
        
        threshold = euclidian_distance(min_d, max_d)/10    
            
        room["relationships"] = []
        for i in range(0, number_of_objects):
            # if objects_info[i]["category"] == "baseboard":
                #     continue
                
            for j in range(i+1, number_of_objects):
                # if objects_info[j]["category"] == "baseboard":
                #     continue
            
                distance = euclidian_distance(objects_info[i]["location"], objects_info[j]["location"])
                
                if distance < threshold:
                    room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 6 , "close by"])
                    room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 6 , "close by"])
                
                if distance < threshold/2:
                    if objects_info[i]["location"][0] < objects_info[j]["location"][0]:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 0 , "left"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 1 , "right"])
                    else:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 1 , "right"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 0 , "left"])
                    
                    
                        
                    if objects_info[i]["location"][1] < objects_info[j]["location"][1]:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 2 , "front"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 3 , "behind"])
                    else:
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 2 , "front"])
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 3 , "behind"])
                    
                    
                    
                    if objects_info[i]["location"][2]+objects_info[i]["dimension"][1]/2 < objects_info[j]["location"][2]+objects_info[j]["dimension"][1]/2:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 9 , "lower than"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 8 , "higher than"])
                    else:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 8 , "higher than"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 9 , "lower than"])
                    
                    
                    
                    if volume(objects_info[i]["dimension"]) <  volume(objects_info[j]["dimension"]):
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 4 , "smaller than"])
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 5 , "bigger than"])
                    else:
                        room["relationships"].append([objects_info[j]["id_sg"], objects_info[i]["id_sg"], 4 , "smaller than"])
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 5 , "bigger than"])
                        
                        
                        
                        
                    lower_part_obj_1 = objects_info[i]["location"][1] + objects_info[i]["dimension"][1]/2 
                    upper_part_obj_2 =  objects_info[j]["location"][1] + objects_info[j]["dimension"][1]/2 
                    if lower_part_obj_1 - upper_part_obj_2 < 0.02:
                        room["relationships"].append([objects_info[i]["id_sg"], objects_info[j]["id_sg"], 7 , "standing on"])
                        
                    
                    
        relationship_json["scans"].append(room)            
               
    if args.val:
        with open(os.path.join(dir, 'relationships_val.json'), 'w') as f:
            json.dump(relationship_json, f)
    else:
        with open(os.path.join(dir, 'relationships_train.json'), 'w') as f:
            json.dump(relationship_json, f)


    with open(os.path.join(OUTPUT_DIR, "3dfront", "processed_data", "vocab", 'relationships.tsv'), 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(["id", "label"])
        tsv_writer.writerow(["0", "left"])
        tsv_writer.writerow(["1", "right"])
        tsv_writer.writerow(["2", "front"])
        tsv_writer.writerow(["3", "behind"])
        tsv_writer.writerow(["4", "smaller than"])
        tsv_writer.writerow(["5", "bigger than"])
        tsv_writer.writerow(["6", "close by"])
        tsv_writer.writerow(["7", "standing on"])
        tsv_writer.writerow(["8", "higher than"])
        tsv_writer.writerow(["9", "lower than"])

if __name__ == "__main__":
    import inspect
    import sys
    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, parentdir)
    from config.paths import OUTPUT_DIR
    
    
    main()
