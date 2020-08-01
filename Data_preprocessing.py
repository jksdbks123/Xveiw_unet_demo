import os
import sys
from pathlib import Path
import imageio
from collections import defaultdict
# Create folders to hold images and masks
from shapely import wkt
import json
import numpy as np
import skimage.draw
import skimage.io
from tqdm.auto import tqdm
damage_codes = {
    "un-classified": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
def json2dict(filename): # input: json url
    with open(filename) as f:
        j = json.load(f)
    return j
def get_disaster_dict(labels_generator,LABEL_PATH):  # Classify the dirs by type of disaster
    disastor_dict = defaultdict(list)
    for label in labels_generator:
        disaster_type = label.name.split('_')[0]
        disastor_dict[disaster_type].append(LABEL_PATH + '/' + label.name)
    return disastor_dict
if __name__ == '__main__':
    os.chdir(r'/Users/czhui960/Documents/Segdataset/test')
    DATA_PATH = os.getcwd()
    IMAGE_PATH = DATA_PATH + r'/images'
    LABEL_PATH = DATA_PATH + r'/labels'
    labels_generator = Path(LABEL_PATH).rglob(pattern=f'*post_*.json')
    disaster_dict = get_disaster_dict(labels_generator,LABEL_PATH)
    # pre_json, pre_img, post_json , post_img
    dict_pre_post_pairs = {}
    for key in disaster_dict.keys():
        pre_post_pairs = [(disaster_dict[key][i],
                           #                        disaster_dict[key][i].replace('labels','targets')[:-5]+'_target.png',
                           disaster_dict[key][i].replace('labels', 'images').replace('json', 'png'),
                           disaster_dict[key][i].replace('post', 'pre'),
                           #                        (disaster_dict[key][i].replace('labels','targets')[:-5]+'_target.png').replace('post','pre'),
                           disaster_dict[key][i].replace('labels', 'images').replace('json', 'png').replace('post',
                                                                                                            'pre'),
                           disaster_dict[key][i].replace('labels', 'masks').replace('.json', '.png')
                           ) for i in range(len(disaster_dict[key]))]
        dict_pre_post_pairs[key] = pre_post_pairs
    os.makedirs(DATA_PATH + '/masks')
    with open('pairs_dict.json', 'w') as outfile:
        json.dump(dict_pre_post_pairs, outfile)
    for key in dict_pre_post_pairs.keys():
        #     print(key,'begin')
        for item in tqdm(range(len(dict_pre_post_pairs[key])), ascii=True, desc=key, file=sys.stdout):
            json_post = json2dict(dict_pre_post_pairs[key][item][0])
            json_pre = json2dict(dict_pre_post_pairs[key][item][2])
            height = json_pre['metadata']['height']
            width = json_pre['metadata']['width']
            if len(json_pre['features']['xy']) == 0:
                mask_full = np.zeros(shape=(height, width)).astype('uint8')
                skimage.io.imsave(dict_pre_post_pairs[key][item][0].replace('labels', 'masks').replace('.json', '.png'),
                                  mask_full, check_contrast=False)
            else:
                mask_full = np.zeros(shape=(height, width)).astype('uint8')
                for build_i in range(len(json_pre['features']['xy'])):
                    polygon = wkt.loads(json_pre['features']['xy'][build_i]['wkt'])
                    polygons_len = len(polygon.exterior.coords)
                    polygon_arr = np.zeros(shape=(polygons_len, 2))
                    label = damage_codes[json_post['features']['xy'][build_i]['properties']['subtype']]
                    for i in range(len(polygon.exterior.coords)):
                        polygon_arr[i, 0], polygon_arr[i, 1] = polygon.exterior.coords[i][0], \
                                                               polygon.exterior.coords[i][1]
                    mask = label * skimage.draw.polygon2mask((height, width), polygon_arr).astype("uint8").T
                    mask_full += mask
                mask_full = mask_full.astype('uint8')
                skimage.io.imsave(dict_pre_post_pairs[key][item][0].replace('labels', 'masks').replace('.json', '.png'),
                                  mask_full, check_contrast=False)


