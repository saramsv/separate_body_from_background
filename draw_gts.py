#python3 generate_annotated_images.py polygon_annotated_tags  where-the-images-are
#while read name; do `wget $name`; done < poly_tagged.20190312.img.paths I did this on my local computer while hacing access to localhost:3000 to get the images
from shapely.geometry import Point
from shapely.geometry import Polygon
import json
import pandas as pd
import sys
import cv2
import os
import numpy as np
import random
import argparse
random.seed(9001)


parser = argparse.ArgumentParser()
parser.add_argument("--tag_info", type = str)
parser.add_argument("--imgs_path", type = str)
parser.add_argument("--n_classes", type = int)
parser.add_argument("--width", type = int)
parser.add_argument("--height", type = int)


args = parser.parse_args()

n_classes = args.n_classes
img_width = args.width
img_height = args.height
imgs_path = args.imgs_path
tag_info = args.tag_info

df = pd.read_csv(tag_info)
img_src = imgs_path

new_size = img_width # it means the images will be 224*224

def read_img(path):
    flag = True
    if os.path.isfile(path) == False:
        print("this image does not exist:" , path)
        flag = False
    img_obj = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    return img_obj, flag

def tag_index(df):
    tags = df['tag']
    tags = map(lambda x:x.lower(), tags) # This is a list of all of the tags but all lowercase
    tags = list(set(tags)) #remove the duplicates by converting to set and convert back to a list
    tag_dic = {t:1 for i, t in enumerate(tags)} #this would only work when bg vs body
    return tag_dic


classes = tag_index(df) # assigning a class number to each class
#classes = {'foot': 1, 'neck': 1, 'body': 1, 'arm': 1, 'scale': 0, 'skeletonized mandible': 1, 'advanced decomposition; hand': 1, 'body; maggots': 1, 'body; mummification': 1, 'leg': 1, 'bg': 0, 'bone': 1, 'legs': 1, 'torso': 1, 'adipocere': 0, 'hand': 1, 'scavenging': 1, 'leg;mummification': 1, 'mummification; leg': 1, 'mold': 1, 'head': 1}
print(classes)
images_seen = set()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

print(colors)

for index, row in df.iterrows():
    # get the image name and object
    path = row['image']
    name = path[path.find('Photos/') + len('Photos/') : ]
    if name not in images_seen: 
        images_seen.add(name)
        name = os.path.join(img_src , name)
        img, flag = read_img(name)

        #get image size and do the resizing
        if flag:
            height, width = img.shape[:2]
            print(width, height)
            if  width <= height:
                width_percent = new_size / float(width)
                new_height = int(float(height) * float(width_percent))
                new_width = new_size
                img = cv2.resize(img, (new_height, new_width))
            elif height <= width:
                height_percent = new_size /float(height)
                new_width = int(float(width) * float(height_percent))
                new_height = new_size
                img = cv2.resize(img, (new_height, new_width)) 

            ann_img = np.zeros((img.shape[1],img.shape[0], 3)).astype('uint8') #create an empty image
            ann_img[:,:,:] = colors[0] # Have the background to be the first color
            height, width = ann_img.shape[:2]
            #Find all of the tags for this image
            print(width, height)
            df_sub = df[df['image'] == path]

            polygons = []

            for index2 , row2 in df_sub.iterrows():
                location = row2['location']
                loc = json.loads(location)
                geometry = loc[0]['geometry'] #get the whole geomety section od the coordinate
                geometry_points = geometry['points']# get the points of the geometry. This is a list of points(x, y) = [{}, {}, ...]
                polygon_points = [] #this will hold the points that shape the polygon for us
                class_id = classes[row2['tag']]
                for p in geometry_points: # access each point to convert it from ratio to numbers 
                    x = p['x'] # x is ratio and needs to be converted to actual number
                    x = x * width
                    y = p['y']#y is ratio and needs to be converted to actual number
                    y = y * height
                    polygon_points.append((x, y))
                polygon = Polygon(polygon_points)
                polygons.append((class_id, polygon))

            #for each pixel:
            for h in range(height):
                for w in range(width):
                    for class_id, polygon in polygons:
                        p = Point(w, h)
                        if polygon.contains(p):
                            ann_img[h,w] = colors[class_id]
                            break


            name2 = name.replace('.JPG', "")
            ann_img = cv2.resize(ann_img, (img_width, img_height))
            cv2.imwrite( name2 + ".png"  ,ann_img )
