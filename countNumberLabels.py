'''Basic file counting the number of labels. We subsequently found a much easier way 
of doing so.
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


def count_labels(label_list):
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
		if counter == 0: #If the label isn't already in the list, add it
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

	list_of_all_labels = count_labels(whale_labels)

	file = open("label_files/list_of_all_labels.txt", "w")
	file.write(str(list_of_all_labels))

	print("List of all labels = " + str(list_of_all_labels))
	print("Number of labels in whale_labels = " + str(len(list_of_all_labels)))


