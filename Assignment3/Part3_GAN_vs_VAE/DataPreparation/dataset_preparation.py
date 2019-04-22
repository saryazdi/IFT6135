# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import numpy as np
import os
import scipy.io


def SVHN2numpy(data_dir):
	'''
	Reads SVHN dataset from .mat file and saves it as .npy
	'''
	filenames = {'X_train': 'train_32x32.mat',
	'X_test': 'test_32x32.mat'}

	dataset = {}
	for setname, filename in filenames.items():
		data_mat = scipy.io.loadmat(f'{data_dir}{filename}')
		X = np.transpose(data_mat['X'], (3, 2, 0, 1)).astype(float)
		y = np.squeeze(data_mat['y']).astype(float)
		dataset[setname] = X
		dataset[setname.replace('X', 'y')] = y
	return dataset

def get_SVHN_dataset(data_dir, validation_split=0.2, seed=None, zero_mean=True, normalize=True):
	'''
	'''
	assert validation_split < 1
	if seed is not None: np.random.seed(seed)
	X_train_max = None

	try:
		print('Generating numpy dataset from existing .mat files in ' + data_dir)
		dataset = SVHN2numpy(data_dir)
		train_dataset = dataset['X_train']
		train_labels = dataset['y_train']
		X_test = dataset['X_test']
		y_test = dataset['y_test']
	except EnvironmentError:
		raise ValueError('Invalid data_dir: ' + data_dir +
			'. data_dir should point to the parent directory of "train_32x32.mat"\
			and "test_32x32.mat" files.')
	
	# Split train and validation set
	len_train = len(train_labels)
	len_val = int(validation_split * len_train)
	ind = np.arange(len_train)
	np.random.shuffle(ind)

	X_train = train_dataset[ind[len_val:]]
	y_train = train_labels[ind[len_val:]]
	X_val = train_dataset[ind[:len_val]]
	y_val = train_labels[ind[:len_val]]

	if normalize:
		# Normalize the data by 255
		X_train_max = 255
		X_train /= X_train_max
		X_val /= X_train_max
		X_test /= X_train_max

	if zero_mean:
		# zero-mean the data and scale so its [-1, 1]
		X_train = (X_train - 0.5) * 2
		X_val = (X_val - 0.5) * 2
		X_test = (X_test - 0.5) * 2


	return X_train, y_train, X_val, y_val, X_test, y_test, X_train_max