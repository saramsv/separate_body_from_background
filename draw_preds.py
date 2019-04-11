#Running this script  python predict.py --save_weights_path="weights/vgg_segnet-th-Aug-036-0.908.hdf5" --epoch_number=0 --test_images="../mAP/mAPImages/" --output_path="data/mAPImgsBG/" --n_classes=2 --input_height=224 --input_width=224 --model_name="vgg_segnet"

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

random.seed(9001)


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

# To create a path for original images with the background being removed

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
print(colors)
for imgName in images:
        outName = imgName.replace( images_path ,  args.output_path )
        img = cv2.imread(imgName, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
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







