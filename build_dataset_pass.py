"""Split the whales dataset into train and dev sets.

The whales dataset comes in the following format:
    train_whales/
        0_IMG_5864.jpg
        ...

Original images have varying sizes, but generally a width of around 1050 pixels.
Resizing to (256, 256) will give us uniform image sizes and hopefully preserve enough 
resolution to distinguish individual whales. In this first attempt, we are simply resizing without
special attention to aspect ratio, which will lead to image distortion.

We already have a test set created, so we only need to split "train_whales" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set to be as
representative as possible, we'll take 20% of "train_whales" as dev set.
"""

import argparse
import random
import os
import keras.preprocessing.image as kpi

from PIL import Image
from tqdm import tqdm


height = 256
width = 256
SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/renamed_whales', help="Directory with the whales dataset")
parser.add_argument('--int_dir', default='data/augmented_whales', help='Intermediate file before resizing and recoloring')
parser.add_argument('--output_dir', default='data/processed_whales', help="Where to write the new data")

num_images = len(os.listdir(train_data_dir))

def make_grey(filename):
    image = Image.open(filename)
    #Converting RGB images to greyscale
    image = image.convert("L")
    #image.save(filename)   #Uncomment if not in a pipeline with rotate, shift, zoom


def rand_rotate(filename, int_dir):
    num_rotations = 4
    total_rotations = num_rotations * num_images

    random.seed(144)
    rotation_angles = np.random.uniform(0, 45, total_rotations)

    #image = Image.open(filename)   #Uncomment if not in a pipeline with make_grey
    for i in range(num_rotations):
        rot_im = image.rotate(rotation_angles[j])
        rot_im.save(os.path.join(int_dir, filename + 'rot' + i))
        j += 1


def data_augment(filename, data_dir, int_dir):
    j = 0
    k = 0

    # Defining the data directory
    aug_data_dir = os.path.join(args.data_dir, 'renamed_whales')
    # Getting the filenames in the directory
    filenames = [os.path.join(aug_data_dir, f) for f in filenames if f.endswith('.jpg')]

    for filename in tqdm(filenames):

    make_grey(filename)
    rand_rotate(filename, int_dir)
    rand_shift(filename, int_dir)




def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((height, width), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.int_dir, 'augmented_whales')
    #test_data_dir = os.path.join(args.data_dir, 'test_whales')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    #test_filenames = os.listdir(test_data_dir)
    #test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]



    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 #'test': test_filenames
                 }

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev']:
        output_dir_split = os.path.join(args.output_dir, '{}_whales'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
