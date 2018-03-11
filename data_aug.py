""" Augment the dataset by applying RGB to greyscale transformations on the color images,
followed by random rotations, crops, zooms, and shifts

"""

import numpy as np
import keras
import tensorflow as tf 
import argparse
import random
import os
import keras.preprocessing.image as kpi

from PIL import Image
from tqdm import tqdm
from keras_master.keras.preprocessing.image import random_rotation, random_shift, random_zoom

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/renamed_whales', help="Directory with the whales dataset")
parser.add_argument('--int_dir', default='data/augmented_whales', help='Intermediate file before resizing and recoloring')
#parser.add_argument('--output_dir', default='data/processed_whales', help="Where to write the new data")


def data_augment(data_dir, int_dir):
# data_augment takes each image from the data_dir, converts it to grey scale,
# and then generates a set of manipulated images from it. Each manipulation is performed independently
# to create a new file; i.e., a zoom and shift are not combined to generate one new image.

	# Defining the data directory
    aug_data_dir = args.data_dir # os.path.join(args.data_dir, 'renamed_whales')

    # Getting the filenames in the directory
    filenames = os.listdir(aug_data_dir)
    filenames = [os.path.join(aug_data_dir, f) for f in filenames if f.endswith('.jpg')]

    ### Define rotation criteria ###
    # Number of rotations per image
    num_rotations = 3
    # Random rotation range in degrees
    rot_range = 45
    # Axis indices
    row_axis = 0
    col_axis = 1
    channel_axis = 2


    ### Define shift criteria ###
    # Number of shifts per image
    num_shifts = 2
    # Width shift range, as float fraction of image width
    wrg = 0.2
    # Height shift range, as float fraction of image width
    hrg = 0.2
    # Channel axes: same as for rotation


    ### Define random zoom criteria ###
    # Number of random zooms per image
    num_zooms = 1
    # Zoom range: tuple of floats for width and height
    zoom_range_width = 2
    zoom_range_height = 2


    for filename in tqdm(filenames):
    	image = Image.open(filename)
    	# Ensuring all images are RGB (3D array):
    	image = image.convert('RGB')
    	# Convert image to array:
    	image = np.array(image)

    	# Randomly rotate image:
    	for i in range(num_rotations):
    		rot_im = random_rotation(image, rot_range, row_axis = row_axis, col_axis = col_axis, channel_axis = channel_axis,
    			fill_mode = 'nearest')
    		# Convert back to image
    		rot_im = Image.fromarray(rot_im, 'RGB')
    		# Save each randomly rotated image in the int_dir
    		rot_im.save(os.path.join(int_dir, ((filename).split('/')[-1]).split('.')[0]) + '_rot' + str(i) +'.jpg', format = 'jpeg')

    	# Randomly shift image:
    	for i in range(num_shifts):
    		shift_im = random_shift(image, wrg = wrg, hrg = hrg, row_axis = row_axis, col_axis = col_axis,
    			channel_axis = channel_axis, fill_mode = 'nearest')
    		# Convert back to image
    		shift_im = Image.fromarray(shift_im, 'RGB')
    		# Save each randomly shited image to the int_dir
    		shift_im.save(os.path.join(int_dir, ((filename).split('/')[-1]).split('.')[0]) + '_shift' + str(i) +'.jpg', format = 'jpeg')

    	# Randomly zoom image:
    	for i in range(num_zooms):
    		zoom_im = random_zoom(image, zoom_range = (zoom_range_width, zoom_range_height), row_axis = row_axis, col_axis = col_axis,
    			channel_axis = channel_axis, fill_mode = 'nearest')
    		# Convert back to image
    		zoom_im = Image.fromarray(zoom_im, 'RGB')
    		# Save each randomly zoomed image to the int_dir
    		zoom_im.save(os.path.join(int_dir, ((filename).split('/')[-1]).split('.')[0]) + '_zoom' + str(i) +'.jpg', format = 'jpeg')



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if not os.path.exists(args.int_dir):
        os.mkdir(args.int_dir)
    else:
        print("Warning: int dir {} already exists".format(args.int_dir))

    data_augment(args.data_dir, args.int_dir)

    print("Done building dataset")




