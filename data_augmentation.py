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


height = 256
width = 256
SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/renamed_whales', help="Directory with the whales dataset")
parser.add_argument('--int_dir', default='data/augmented_whales', help='Intermediate file before resizing and recoloring')
#parser.add_argument('--output_dir', default='data/processed_whales', help="Where to write the new data")


"""
def make_grey(filename):
    image = Image.open(filename)
    #Converting RGB images to greyscale
    image = image.convert("L")
    #image.save(filename)   #Uncomment if not in a pipeline with rotate, shift, zoom


def rand_rotate(filename, int_dir):
    num_rotations = 4
    #image = Image.open(filename)   #Uncomment if not in a pipeline with make_grey
    for i in range(num_rotations):
        rot_im = image.rotate(rotation_angles[j])
        rot_im.save(os.path.join(int_dir, filename + 'rot' + i))
        j += 1
"""

def data_augment(data_dir, int_dir):
    j = 0
    k = 0

    # Defining the data directory
    aug_data_dir = os.path.join(args.data_dir, 'renamed_whales')
    # Getting the filenames in the directory
    filenames = os.listdir(aug_data_dir)
    filenames = [os.path.join(aug_data_dir, f) for f in filenames if f.endswith('.jpg')]

    # Defining number of images to calcuate size of random rotation and shift arrays
    num_rotations = 4
    num_images = len(filenames)
    # Total number of rotations for set
    total_rotations = num_rotations * num_images

    random.seed(144)
    # Creating a vector of random rotation values from -45 to 45 degrees
    rotation_angles = np.random.uniform(-45, 45, total_rotations) 

    for filename in tqdm(filenames):

    	#make_grey(filename):
    	image = Image.open(filename)
    	#Converting RGB images to greyscale
    	image = image.convert("L")
    	
    	#rand_rotate(filename, int_dir)
    	for i in range(num_rotations):
        	rot_im = image.rotate(rotation_angles[j])
        	#rot_im.save(int_dir + filename + 'rot' + str(i))
        	rot_im.save(os.path.join(int_dir, ((filename).split('/')[-1]).split('.')[0]) + '_rot' + str(i) +'.jpg', format = 'jpeg')
        	j = j + 1
         
    	#rand_shift(filename, int_dir)



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if not os.path.exists(args.int_dir):
        os.mkdir(args.int_dir)
    else:
        print("Warning: int dir {} already exists".format(args.int_dir))

    data_augment(args.data_dir, args.int_dir)

    print("Done building dataset")










