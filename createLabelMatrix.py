'''Creates the file that summarizes each unique label, "labels_matrix_nonw.txt", 
WITHOUT including the 'new_whale' label. 

The output file has 4250 rows which correspond to the 4250 classes of whales except 'new whale'
'''

import argparse
import random
import os
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--whale_labels_file', default='label_files/whale_labels.txt', help="Where are your labels?")

def print_labels_matrix(labels_matrix): 
	file = open("label_files/labels_matrix_nonw.txt", "w") # Replace by "labels_matrix.txt otherwise"
	for i in range(len(labels_matrix)):
		file.write(str(labels_matrix[i]+'\n'))

def create_labels_matrix(label_list):
	'''Calls for the renaming of the files with the convention {label}_IMG_{id}.jpg
	Takes as input the label_list, vector [[name of the file, label]]
	'''

	label_list_count = []
	'''For each image name, explore all the labels from the label Rename the image'''
	for i in range(label_list.shape[0]):
		counter = 0
		for j in range(len(label_list_count)):
			if label_list[i][1] == label_list_count[j]:
				counter += 1
		if counter == 0 and label_list[i][1] != 'new_whale': #If the label isn't already in the list, add it
			label_list_count.append(label_list[i][1])
			#print("len(label_list_count) =" + str(len(label_list_count)))

	return label_list_count
		
	'''Copy file on the new directory, then rename it'''
	#rename_and_save_image(dirname, label, data_dir, output_dir)

if __name__ == '__main__':
	args = parser.parse_args()

	whale_labels = np.loadtxt(args.whale_labels_file, delimiter = ',', dtype="U25")

	##Gets the filenames from the directory
	#filenames = os.listdir(args.data_dir)
	#filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.jpg')]

	list_of_all_labels = create_labels_matrix(whale_labels)

	print_labels_matrix(list_of_all_labels)

	print("List of all labels = " + str(list_of_all_labels))
	#print("Number of labels in whale_labels = " + str(len(list_of_all_labels)))


