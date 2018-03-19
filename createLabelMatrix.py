import argparse
import random
import os
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--whale_labels_file', default='whale_labels.txt', help="Where are your labels?")

def print_labels_matrix(labels_matrix): 
	file = open("labels_matrix_nonw.txt", "w") # Replace by "labels_matrix.txt otherwise"
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

	'''FORMER TEST dt = np.dtype([('label', np.unicode_, 16), ('label_number', np.dtype('i4'))])
		label_unique_list = np.empty([4251,1],dtype=dt)
		print("label_unique_list = "+str(label_unique_list))
		print("label_unique_list ['label']= "+str(label_unique_list['label']))'''

	''' FORMER TEST
	k = 0 # Index at which we write the new labels
	for i in range(label_list.shape[0]):
		print(i)
		counter = 0
		for j in range(len(label_list_count)):
			if label_list[i][1] == label_unique_list[j]['label']:
				counter += 1
		if counter == 0: #If the label isn't already in the list, add it
			#label_list_count.append(label_list[i][1])
			label_unique_list[k]['label']=label_list[i][1]
			label_unique_list[k]['label_number']=k
			k += 1
			print("k ="+str(k))
			#print("len(label_list_count) =" + str(len(label_list_count)))
	'''
	return label_list_count
		
	'''Copy file on the new directory, then rename it'''
	#rename_and_save_image(dirname, label, data_dir, output_dir)

if __name__ == '__main__':
	args = parser.parse_args()

	whale_labels_essai = np.array([["0a0b2a01.jpg","1"],["0a0bc259.jpg","2"],["0a00c8c5.jpg","3"]])
	#print("whale_labels_essai[1] = " +str(whale_labels_essai[1]))

	#print("args.whale_labels_file ="+str(args.whale_labels_file))
	'''print("pd.read_csv(args.whale_labels_file, sep =',', header=None)="+str(pd.read_csv(args.whale_labels_file, sep =',', header=None)))
	whale_labels = pd.read_csv(args.whale_labels_file, sep =',', header=None).as_matrix() '''#Reads then convert into numpy array
	
	whale_labels = np.loadtxt(args.whale_labels_file, delimiter = ',', dtype="U25")

	#print("whale_labels = " + str(whale_labels))

	##Gets the filenames from the directory
	#filenames = os.listdir(args.data_dir)
	#filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.jpg')]

	list_of_all_labels = create_labels_matrix(whale_labels)

	print_labels_matrix(list_of_all_labels)

	print("List of all labels = " + str(list_of_all_labels))
	#print("Number of labels in whale_labels = " + str(len(list_of_all_labels)))


