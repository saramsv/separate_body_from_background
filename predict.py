#Running this script  python predict.py --save_weights_path="weights/vgg_segnet-th-Aug-036-0.908.hdf5" --epoch_number=0 --test_images="../mAP/mAPImages/" --output_path="data/mAPImgsBG/" --n_classes=2 --input_height=224 --input_width=224 --model_name="vgg_segnet" --on_original_imgs="true"

# This will create a mask for each image which shows background vs wvey other classes
# Also it create a directory that include the actuall images with their baclground being replaces with color black
import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import os
import numpy as np
import random

def apply_on_orig_img(path, mask, outName, odering='channels_first' ):

    try:
        img = cv2.imread(path, 1)
	orig_w = img.shape[1]
	orig_h = img.shape[0]

        mask = cv2.resize(mask, (orig_w, orig_h)) # resizing the mask to be the same size as the original images
        black_mask = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0)) # what pixels are black = all 3 channels
        black_indices = zip(black_mask[0], black_mask[1]) # creating tuples from x and y indices
        for coor in black_indices: #setting the color of the coresponding pixels in the original image as black(0, 0, 0)
            img[coor[0], coor[1], 0] = 0
            img[coor[0], coor[1], 1] = 0
            img[coor[0], coor[1], 2] = 0
        cv2.imwrite(outName, img)

    except Exception, e:
        print path , e 
        img = np.zeros((  orig_h , orig_w  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )
parser.add_argument("--on_original_imgs", type = str, default = "false")

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
on_orig_imgs = args.on_original_imgs

# To create a path for original images with the background being removed
bg_sep_imgs_path = args.output_path + "/on_orig_imgs/" 
if on_orig_imgs == "true":
    if not os.path.exists(bg_sep_imgs_path):
        os.makedirs(bg_sep_imgs_path)

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
#m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
##Sara
m.load_weights(  args.save_weights_path)
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.JPG"  ) +  glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

not_bg_color = [(255, 255, 255)] # eveything other than background

for imgName in images:
        if on_orig_imgs == 'false':
            outName = imgName.replace( images_path ,  args.output_path )
            X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
            pr = m.predict( np.array([X]) )[0]
            pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
            seg_img = np.zeros( ( output_height , output_width , 3  ) )
            for c in range(n_classes):
                    seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
                    seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
                    seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
            seg_img = cv2.resize(seg_img  , (input_width , input_height ))
            cv2.imwrite(  outName , seg_img )

        elif on_orig_imgs == "true":
            outName = imgName.replace( images_path ,  args.output_path )
            X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
            pr = m.predict( np.array([X]) )[0]
            pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
            seg_img = np.zeros( ( output_height , output_width , 3  ) )
            for c in range(1): #only the background and then we color everything other than the bg with white
                    seg_img[:,:,0] += ( (pr[:,: ] != c )*( not_bg_color[c][0] )).astype('uint8')
                    seg_img[:,:,1] += ((pr[:,: ] != c )*( not_bg_color[c][1] )).astype('uint8')
                    seg_img[:,:,2] += ((pr[:,: ] != c )*( not_bg_color[c][2] )).astype('uint8')
            seg_img = cv2.resize(seg_img  , (input_width , input_height )) # the same size as the resized_image
            cv2.imwrite(  outName , seg_img )
            outName = imgName.replace( images_path ,  bg_sep_imgs_path )
            apply_on_orig_img(imgName, seg_img, outName)
            







