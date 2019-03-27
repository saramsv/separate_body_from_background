from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import sys
import cv2

img_path = sys.argv[1]
ann_path = sys.argv[2]

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format = 'channels_last')
count = 0
for img in sorted(glob.glob(img_path + "*.JPG")):
    img = load_img(img)
    x = img_to_array(img) #numpy array of shape(3, x, y)
    x = x.reshape((1, ) + x.shape)
    #x = cv2.imread(img)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='test/', shuffle=False, save_prefix=str(count), save_format='png', seed = 1):
        i += 1
        if i > 4:
            break  # otherwise the generator would loop indefinitely
    count += 1
count = 0
for img in sorted(glob.glob(ann_path + "*.png")):
    img = load_img(img)
    x = img_to_array(img) #numpy array of shape(3, x, y)
    x = x.reshape((1, ) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='test2/', shuffle=False, save_prefix=str(count), save_format='png', seed = 1):
        i += 1
        if i > 4:
            break  # otherwise the generator would loop indefinitely
    count += 1
