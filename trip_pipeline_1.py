

import numpy as np
import keras
import tensorflow as tf 
import argparse
import random
import os
import keras.preprocessing.image as kpi
import glob
from tensorflow.contrib.losses.python.metric_learning.metric_loss_ops import triplet_semihard_loss

from PIL import Image
from tqdm import tqdm
from keras_master.keras.preprocessing.image import random_rotation, random_shift, random_zoom

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/renamed_whales', help="Directory with the whales dataset")
parser.add_argument('--out_dir', default='data/augmented_whales', help='Output image array directory')



 # Defining the data directory
images = os.path.join(args.data_dir, 'renamed_whales')
# Getting the filenames in the directory
filenames = os.listdir(images)
filenames = [os.path.join(images, f) for f in filenames if f.endswith('.jpg')]

# Convert every file to greyscale
for filename in tqdm(filenames):
	image.open(filename)
	image.convert("L")


# Convert to numpy array 
image_array = np.array([np.array(Image.open(fname)) for fname in filelist])
tf.convert_to_tensor(image_array, name = image_tensor)


"""The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.

  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.

  Returns:
    triplet_loss: tf.float32 scalar. """

metric_loss_ops.triplet_semihard_loss(image_array, squared = False)