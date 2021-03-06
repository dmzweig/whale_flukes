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

from PIL import Image
from tqdm import tqdm


SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/NUMBER_LABELS_WHALES_NONW', help="Directory with the RENAMED_WHALES dataset")
parser.add_argument('--output_dir', default='data/64x64_NUMBER_LABELS_WHALES_NONW', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_whales')
    test_data_dir = os.path.join(args.data_dir, 'test_whales')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

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
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_whales'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
