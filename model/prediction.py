"""Tensorflow utility functions for prediction"""

import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np

from model.utils import save_dict_to_json, save_array_to_txt

def search_4_highest_probabilities(probability_array):
    """Finds the 5 highest probabilities in a 1-dimensional array

    Args:
        probability_array: 1-D array
    Returns: list of 5 highest probabilities
    """
    lst = sorted( [(x,i) for (i,x) in enumerate(probability_array)][:4], reverse=True)
    # print("lst =" +str(lst))

    return lst

def write_predictions_on_txt(filenames, label_matrix_file, predictions, probabilities, txt_path):
    """Saves dict of floats in json file

    Args:
        filenames: array of the directory of all the files
        label_matrix_file: .txt file directory containing the list of the labels
        d: array of float-castable values (np.float, int, float, etc.)
        txt_path: (string) path to json file
    """
    t = open(label_matrix_file, 'r')
    label_matrix = np.loadtxt(t, dtype="U25")

    with open(txt_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        f.write("Image,Id"+'\n')
        for i in range(len(predictions[0])):
            # Extract the 5 highest probabilities (proba, index)
            five_proba_and_labels = search_4_highest_probabilities(probabilities[0][i])
            # print("five_proba_and_labels = "+str(five_proba_and_labels))

            f.write(os.path.basename(os.path.normpath(filenames[i]))+',')
            f.write("new_whale ")
            for (index,label) in five_proba_and_labels:
                for j in range(4250):
                    if label == j:   
                        f.write(str(label_matrix[j])+" ")
                        f.write(str(label)+" ") # Temporary, just to visualize similarities
            f.write('\n')
            

def probabilities_sess(sess, model_spec, num_steps, writer=None, params=None):
    """ 
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """    

    probabilities = model_spec['probabilities']
    global_step = tf.train.get_global_step()

    probabilities_list=[] # Initiate the list of predictions

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    # Dontù need this: sess.run(model_spec['metrics_init_op'])

    # predict over the dataset
    for _ in range(num_steps):
        probabilities_list.append(sess.run(probabilities))

    return probabilities_list

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

    # predict over the dataset
    for _ in range(num_steps):
        predictions_list.append(sess.run(predictions))

    return predictions_list


def predict(model_spec, model_dir, test_filenames, label_matrix_file, params, restore_from):
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
        probabilities = probabilities_sess(sess, model_spec, num_steps)
        # print("probabilities = "+str(probabilities))
        predictions = predict_sess(sess, model_spec, num_steps)
        # print("predictions = "+str(predictions))
        predictions_name = '_'.join(restore_from.split('/'))
        # print("predictions_name = "+str(predictions_name))
        save_path = os.path.join(model_dir, "predictions_test_{}.txt".format(predictions_name))
        write_predictions_on_txt(test_filenames, label_matrix_file, predictions, probabilities, save_path)