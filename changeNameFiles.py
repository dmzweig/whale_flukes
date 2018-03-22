'''Rename the orginal files with their labels
To be run before reducing the size to 64x64'''

import argparse
import random
import os
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='whales_originalfiles/train_whales_original', help="Directory with the WHALES dataset")
parser.add_argument('--output_dir', default='data/NUMBER_LABELS_WHALES_NONW/train_whales', help="Where to write the new data")
parser.add_argument('--whale_labels_file', default='label_files/whale_labels.txt', help="Where are your labels?")

def rename_and_save_image(file_name, label, data_dir, output_dir):
	'''Rename one single file with the convention {label}_IMG_{id}.jpg'''
	shutil.copyfile(os.path.join(data_dir,file_name),os.path.join(output_dir,file_name))
	os.rename(output_dir+"/"+file_name, output_dir +"/"+label+"_IMG_"+file_name)

def rename_and_save(label_list, data_dir, output_dir):
	'''Calls for the renaming of the files with the convention {label}_IMG_{id}.jpg
	Takes as input the label_list, vector [[name of the file, label]]
	'''

	'''For each image name, explore all the labels from the label Rename the image'''
	for dirname in os.listdir(data_dir):
		#print("label_list.shape[0]="+str(label_list.shape[0]))
		for i in range(label_list.shape[0]):
			#print("label_list.shape[i] =" + str(label_list[i]))
			if label_list[i][0] == dirname:
				rename_and_save_image(dirname, label_list[i][2], data_dir, output_dir)

		'''if os.path.isdir(dirname):
			for i, filename in enumerate(os.listdir(dirname)):'''
		
		'''Copy file on the new directory, then rename it'''
		#rename_and_save_image(dirname, label, data_dir, output_dir)

def associate_label_with_number(file, label_list):

	'''Duplicates the file with labels, and add an empty third column full of zeros.
	Each label will be associated with a number
	label_numbers_list is an array [IMG_ID, label, label_number]'''

	temp_array=np.zeros((label_list.shape[0],1))
	label_number_list = np.append(label_list, temp_array, axis = 1)
	
	# counter is used to follow the number of labels. At the end, counter
	# should be equal to 4251
	counter = 0
	for i in range(label_list.shape[0]):
		indicator = 0
		temp_label = 0
		for j in range(i-1):
			if label_number_list[i][1] == 'new_whale':
				temp_label = 5000
				indicator += 1
			else:
				if label_number_list[i][1] == label_number_list[j][1]:
					temp_label = label_number_list[j][2] # If the j label has already been associated with a label number at the i line, and different than new_whale, store the label on temp_label
					indicator += 1
		if (indicator !=0):
			label_number_list[i][2] = temp_label # If the j label has already been associated with a label number, copy the label number on the j line
		else:
			label_number_list[i][2] = counter
			counter += 1 # Otherwise write the new label_number and increase the label index
		file.write(str(label_number_list[i]) + '\n')
		print("counter = "+str(counter))

	print("label_number_list = "+str(label_number_list))

	return label_number_list


if __name__ == '__main__':
	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

	''' Create output folder'''
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	else:
		print("Warning: output dir {} already exists".format(args.output_dir))

	# Open the .txt file on which we will save the whale_labels_numbers: array [IMG_ID, label, label_number]
	# (We need number labels to make the code work)
	# First, we load the file with [IMG_ID, label] into a np.array
	# Then we create the whale_labels_numbers = [IMG_ID, label, label_number]
	file = open("label_files/whale_labels_numbers.txt", "w")
	whale_labels = np.loadtxt(args.whale_labels_file, delimiter = ',', dtype="U25")
	whale_labels_numbers = associate_label_with_number(file, whale_labels)

	# Add the label number at the beginning of the name of each image
	rename_and_save(whale_labels_numbers, args.data_dir, args.output_dir)

	print("Done renaming files and saving them in a new folder")







