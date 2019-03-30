import cv2
import glob
import itertools
from keras.preprocessing.image import ImageDataGenerator


def getImageArr( path , width , height , imgNorm="No_sub_mean" , odering='channels_last' ):
def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):

	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))

		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		elif imgNorm == "sub_mean":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img[:,:,0] -= 103.939
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img = img/255.0
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img
	except Exception, e:
		print path , e
		img = np.zeros((  height , width  , 3 ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img


def getSegmentationImgs(path , nClasses ,  width , height):



def getSegmentationArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
                img[np.where(img[:,:,0] != 0)] = 1 ## This is becuase in my augmentation the 1 have been converted to 255 and some other numbers!
		img = img[:, : , 0]
                #img = np.reshape(img, (width, height,1))# an image with 3 channel, channel_last 
                #img = np.reshape(img, (1, width, height))# an image with 3 channel 

		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception, e:
		print e

	##seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
        #print("seg size:", seg_labels.shape)
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels
	#return img

def getSegmentationArr(imgs , nClasses ,  width , height):
        labels = []
        for img in imgs:
            '''
            seg_labels = np.zeros((  height , width  , nClasses ))
            try:
                    for c in range(nClasses):
                        seg_labels[: , : , c ] = (img[:, :, 0] == 1 ).astype(int)#because everything other than bg is set as 1.
            except Exception, e:
                    print e
                    
            ##seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
            ##labels.append(seg_labels)
            labels.append(seg_labels[:,:,0])
            '''
            labels.append(np.reshape(img, ( width*height , nClasses )))
        return np.array(labels)



def show_img(imgs, name, i):
        for img in imgs:
            #img2 = np.rollaxis(img, 2, 0)
            #img2 = np.rollaxis(img2, 2, 0)
            if name == "img":
                #print("img size:", img.shape)
                cv2.imwrite("test/" + name + str(i) + ".jpg", img)
            if name == "mask":
                #img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                #print("mask size: ",img[:,:,0].shape)
                cv2.imwrite("test/" + name + str(i) + ".png",img[:,:,0] )



def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	images = glob.glob( images_path + "*.JPG"  ) + glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()
	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	#zipped = itertools.cycle( zip(images,segmentations) )
	zipped = itertools.cycle( zip(images,segmentations) )

        ##Sara
	zipped = zip(images,segmentations)
        X_train = []
        Y_train = []
        for im_ann in zipped:
            im , seg = im_ann[:2]
            X_train.append(getImageArr(im , input_width , input_height))
            Y_train.append(getSegmentationImgs( seg , n_classes , output_width , output_height))
        #print(np.array(X_train).shape)
        #print(np.array(Y_train).shape)
        return np.array(X_train), np.array(Y_train)

        ## end Sara

        '''
	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = zipped.next()
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationImgs( seg , n_classes , output_width , output_height )  )
                print(np.array(X).shape)
                print(np.array(Y).shape)
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )
                '''    
		x = np.array(X)
		y = np.array(Y)
		i = 0
		for x_batch in datagen.flow(x, batch_size = batch_size, save_to_dir='test/', save_prefix='aug', save_format='jpeg', seed = 1):
                    y_batch = datagen.flow(y, batch_size = batch_size, save_to_dir='test/', save_prefix='aug', save_format='jpeg', seed = 1)
                    print(x_batch.shape)
                    #import bpython
                    #bpython.embed(locals())
		    yield x_batch , y_batch
                    #exit()
		    i += 1
		    if i > 0:
			break  # otherwise the generator would loop indefinitely
                '''
		yield np.array(X) , np.array(Y)

        '''

# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )


