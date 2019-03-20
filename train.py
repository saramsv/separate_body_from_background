import argparse
import keras
import os
import Models , LoadBatches
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

parser.add_argument("--epochs", type = int, default = 2)
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

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]


m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
callbacks = [keras.callbacks.TensorBoard(log_dir = save_weights_path), keras.callbacks.ModelCheckpoint(save_weights_path + "{epoch:03d}-{val_acc:.3f}.hdf5", verbose = 0, monitor = 'val_acc', mode = 'max', save_best_only = True)]

loss_ = 'categorical_crossentrop'
if n_classes == 1:
        loss_ = 'binary_crossentropy'

        optimizer_name = optimizers.SGD(lr = 0.001, clipvalue = 0.5, decay = 1e-6, momentum = 0.9, nesterov = True)
        m.compile(loss= loss_,
                      optimizer= optimizer_name ,
                            metrics=['accuracy'])



if len( load_weights ) > 0:
	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth
G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

G_v  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )



if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
    m.fit_generator(
		    G,
		    steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		    epochs=epochs,
		    validation_data=G_v,
		    callbacks = callbacks,
		    validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
else:
    m.fit_generator(
		    G,
		    steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		    epochs=epochs,
		    validation_data=G_v,
		    callbacks = callbacks,
		    validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
