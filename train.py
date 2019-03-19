import argparse
import keras
from keras.callbacks import TensorBoard
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import Models, LoadBatches
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator



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
parser.add_argument("--batch_size", type = int, default = 2 )
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

optimizer_name = optimizers.SGD(lr = 0.001, clipvalue = 0.5, decay = 1e-6, momentum = 0.9, nesterov = True)
m.compile(loss= loss_,
      optimizer= optimizer_name ,
      metrics=['accuracy'])

callbacks = [keras.callbacks.TensorBoard(log_dir = save_weights_path), keras.callbacks.ModelCheckpoint(save_weights_path + "{epoch:03d}-{val_acc:.3f}.hdf5", verbose = 0, monitor = 'val_acc', mode = 'max', save_best_only = True)]

if len( load_weights ) > 0:
	m.load_weights(load_weights)

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


output_height = m.outputHeight
output_width = m.outputWidth

x_train, y_train  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
x_val, y_val  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
# Provide the same seed and keyword arguments to the fit and flow methods
##### train data #########
seed = 1
image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

train_generator = image_datagen.flow(x_train, y_train, batch_size =train_batch_size, seed=seed)
'''
mask_generator = mask_datagen.flow_from_directory(
    'data/train.ann/',
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks for train images
train_generator = zip(image_generator, mask_generator)
'''
############ val data ############
seed = 1
image_datagen.fit(x_val, augment=True, seed=seed)
mask_datagen.fit(y_val, augment=True, seed=seed)
val_generator = image_datagen.flow(x_val, y_val, batch_size = val_batch_size, seed=seed)
'''
image_generator_val = image_datagen.flow_from_directory(
    'data/val/',
    class_mode=None,
    seed=seed)

mask_generator_val = mask_datagen.flow_from_directory(
    'data/val.ann/',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks for val images
val_generator = zip(image_generator_val, mask_generator_val)
'''
m.fit_generator(
	train_data = train_generator,
	steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
	epochs=epochs,
	validation_data=val_generator,
	callbacks = callbacks,
	validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
print("got to the end")
'''
print("Model output shape" ,  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

#G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
#G_v  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
x_train, y_train  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
x_val, y_val  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)


if validate:
	x_val, y_val  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	## Start Sara
	m.fit_generator(
		datagen.flow(x_train, y_train, train_batch_size),
		steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		epochs=epochs,
		validation_data=datagen.flow(x_val, y_val, val_batch_size),
		callbacks = callbacks,
		validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
	## End Sara
	#m.fit_generator( G , 512, epochs= epochs , callbacks = callbacks)
	##for ep in range( epochs ):
	##	m.fit_generator( G , 512  , epochs=1 , callbacks = callbacks)
	##	m.save_weights( save_weights_path + "." + str( ep ) )
	##	m.save( save_weights_path + ".model." + str( ep ) )
else:
	## Start Sara
	m.fit_generator(
		datagen.flow(x_train, y_train, train_batch_size),
		steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		epochs=epochs,
		validation_data=datagen.flow(x_val, y_val, val_batch_size),
		callbacks = callbacks,
		validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
	## End Sara
	#m.fit_generator( G , 512, epochs= epochs , callbacks = callbacks)
	##for ep in range( epochs ):
	##	m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
	##	m.save_weights( save_weights_path + "." + str( ep )  )
	##	m.save( save_weights_path + ".model." + str( ep ) )
'''
