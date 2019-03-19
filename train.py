import argparse
import keras
from keras.callbacks import TensorBoard
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import Models, LoadBatches
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, list_pictures
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 10 )
parser.add_argument("--batch_size", type = int, default = 80 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name
## Start Sara
val_images_path = args.val_images
val_segs_path = args.val_annotations
val_batch_size = args.val_batch_size
## End Sara

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width)
loss_ = 'categorical_crossentrop'
if n_classes == 1:
    loss_ = 'binary_crossentropy'

callbacks = [keras.callbacks.TensorBoard(log_dir = save_weights_path), keras.callbacks.ModelCheckpoint(save_weights_path + "{epoch:03d}-{val_acc:.3f}.hdf5", verbose = 0, monitor = 'val_acc', mode = 'max', save_best_only = True)]

optimizer_name = optimizers.SGD(lr = 0.001, clipvalue = 0.5, decay = 1e-6, momentum = 0.9, nesterov = True)
m.compile(loss= loss_,
      optimizer= optimizer_name ,
      metrics=['accuracy'])

callbacks = [keras.callbacks.TensorBoard(log_dir = save_weights_path), keras.callbacks.ModelCheckpoint(save_weights_path + "{epoch:03d}-{val_acc:.3f}.hdf5", verbose = 0, monitor = 'val_acc', mode = 'max', save_best_only = True)]

if len( load_weights ) > 0:
	m.load_weights(load_weights)

output_height = m.outputHeight
output_width = m.outputWidth

x_train, y_train  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
x_val, y_val  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

'''
import bpython
bpython.embed(locals())
'''
def image_augmentation(imgs, masks): 
    #  create two instances with the same arguments
    # create dictionary with the input augmentation values
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2, 
                         horizontal_flip=True,
                         vertical_flip = True)
    ## use this method with both images and masks
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    ## fit the augmentation model to the images and masks with the same seed
    image_datagen.fit(imgs, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    ## set the parameters for the data to come from (images)
    image_generator = image_datagen.flow(
        imgs,
        batch_size=train_batch_size,
        shuffle=True,
        seed=seed)
    ## set the parameters for the data to come from (masks)
    mask_generator = mask_datagen.flow(
        masks,
        batch_size=train_batch_size,
        shuffle=True,
        seed=seed)
    while True:
        yield(image_generator.next(), mask_generator.next())

    # combine generators into one which yields image and masks
    ##train_generator = zip(image_generator, mask_generator)
    ## return the train generator for input in the CNN 
    #return train_generator

train_generator = image_augmentation(x_train, y_train)
        ## run the fit generator CNN
m.fit_generator(
		train_generator,
		steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		epochs=epochs,
		callbacks = callbacks)
