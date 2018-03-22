"""Predict the model and outputs the 15,000+ predictions under the formatting required
for our Kaggle challenge submission, as a .csv file:

IMG_ID,prediction_1 prediction_2 prediction_3 prediction_4 prediction_5"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn_test
from model.model_fn import model_fn
from model.prediction import predict
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/learning_rate',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_NUMBER_LABELS_WHALES_NONW',
                    help="Directory containing the dataset")
parser.add_argument('--label_matrix', default='label_files/labels_matrix_nonw.txt',  help="Matrix of the labels <> numbers, without new_whale")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_whales")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Get the list of labels
    labels_matrix_nonw_file = args.label_matrix

    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn_test(test_filenames, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('predict', test_inputs, params, reuse=False)

    logging.info("Starting prediction")
    predict(model_spec, args.model_dir, test_filenames, labels_matrix_nonw_file, params, args.restore_from)
