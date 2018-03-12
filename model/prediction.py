"""Tensorflow utility functions for prediction"""

import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np

from model.utils import save_dict_to_json, save_array_to_txt


def predict_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    predictions = model_spec['predictions']
    global_step = tf.train.get_global_step()

    predictions_list=[] # Initiate the list of predictions

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    # Dontù need this: sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    for _ in range(num_steps):
        predictions_list.append(sess.run(predictions))

    return predictions_list


def predict(model_spec, model_dir, params, restore_from):
    """Predict the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Predict
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        predictions = predict_sess(sess, model_spec, num_steps)
        predictions_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "predictions_test_{}.txt".format(predictions_name))
        save_array_to_txt(predictions, save_path)