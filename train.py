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

parser.add_argument("--epochs", type = int, default = 200)
parser.add_argument("--batch_size", type = int, default = 80 )
parser.add_argument("--val_batch_size", type = int, default = 35 )
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
if n_classes == 2:
        loss_ = 'binary_crossentropy'

optimizer_name = optimizers.SGD(lr = 0.01, clipvalue = 0.5, decay = 1e-6, momentum = 0.9, nesterov = True)
m.compile(loss= loss_,
        optimizer= optimizer_name ,
        metrics=['accuracy'])

if len( load_weights ) > 0:
	m.load_weights(load_weights)

print "Model output shape" ,  m.output_shape


output_height = m.outputHeight
output_width = m.outputWidth

#output_height = m.output_shape[1]
#output_width = m.output_shape[2]

'''
G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

G_v  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
'''
x_train, y_train  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

x_val, y_val  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

def image_augmentation(imgs, masks): 
    #  create two instances with the same arguments
    # create dictionary with the input augmentation values
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3, 
                         horizontal_flip=True,
                         vertical_flip = True)
    #data_gen_args = dict()
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
        x = imgs,
        y = None,
        batch_size=1,
        shuffle=False,
        seed=seed)
    ## set the parameters for the data to come from (masks)
    mask_generator = mask_datagen.flow(
        masks,
        batch_size=1,
        shuffle=False,
        seed=seed)
    #LoadBatches.show_img(image_generator.next(), "img")
    #LoadBatches.show_img(mask_generator.next(), "mask")
    #exit()
    i = 0
    while True:
        #print(i)
        img_gen = image_generator.next()
        mask_gen = mask_generator.next()
        #LoadBatches.show_img(img_gen, "img", i )
        #LoadBatches.show_img(mask_gen, "mask", i)
        #i = i+ 1
        yield(img_gen,LoadBatches.getSegmentationArr(mask_gen, n_classes, output_height, output_width))

train_generator = image_augmentation(x_train, y_train)
val_generator = image_augmentation(x_val, y_val)

if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
    m.fit_generator(
		    train_generator,
		    steps_per_epoch=(sum([len(files) for r, d, files in os.walk(train_images_path)])) // train_batch_size,
		    epochs=epochs,
		    validation_data=val_generator,
		    callbacks = callbacks,
		    validation_steps= (sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
else:
    m.fit_generator(
		    train_generator,
		    steps_per_epoch=1000,#(sum([len(files) for r, d, files in os.walk(train_images_path)])) train_batch_size,
		    epochs=epochs,
		    validation_data=val_generator,
		    callbacks = callbacks,
		    validation_steps= 500)#(sum([len(files) for r, d, files in os.walk(val_images_path)])) // val_batch_size)
