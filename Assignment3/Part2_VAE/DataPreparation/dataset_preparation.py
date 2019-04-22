# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import numpy as np
import os
from imageio import imread

def binarymnist(data_dir):
	'''
	Converts binary MNIST dataset from .amat format into .npy
	'''
	filenames = {'X_train': 'binarized_mnist_train.amat',
	'X_val': 'binarized_mnist_valid.amat',
	'X_test': 'binarized_mnist_test.amat'}

	dataset = {}
	for setname, filename in filenames.items():
		with open(data_dir + filename, 'r') as f:
			data = f.readlines()
		data_npy = []
		for x in data:
			x = [int(i) for i in x.split()]
			data_npy.append(np.reshape(np.asarray(x), (1, 28, 28)))
		data_npy = np.array(data_npy, dtype='uint8')
		np.save(f'{data_dir}{setname}', data_npy)
		print('Saving ' + setname + ' done...')

def get_binarymnist_dataset(data_dir, normalize=False):
	'''
	Loads binary MNIST dataset. If numpy files exist in data_dir, loads them.
	Else, it will create numpy files from the .amat files in data_dir using
	binarymnist(data_dir) function.

	Inputs:
	- data_dir: should contain EITHER .amat files of binary MNIST dataset
	OR X_train.npy, X_val.npy and X_test.npy.
	- normalize_train: Whether to normailze the training set or not.

	Returns:
	- X_train: Training images.
	- X_val: Validation images.
	- X_test: Test images.
	- X_train_moments: Tuple of mean and standard deviation of X_train. (Used for
	visualizing original samples)
	'''
	X_train_moments = None, None
	try: # Try to load numpy dataset
		X_train = np.load(f'{data_dir}X_train.npy').astype('float32')
		X_val = np.load(f'{data_dir}X_val.npy').astype('float32')
		X_test = np.load(f'{data_dir}X_test.npy').astype('float32')
	except EnvironmentError: # If numpy dataset doesn't exist, convert jpg data to numpy
		try:
			print('Generating numpy dataset from existing .amat files in "' + data_dir)
			binarymnist(data_dir)
			X_train = np.load(f'{data_dir}X_train.npy').astype('float32')
			X_val = np.load(f'{data_dir}X_val.npy').astype('float32')
			X_test = np.load(f'{data_dir}X_test.npy').astype('float32')
		except EnvironmentError:
			raise ValueError('Invalid data_dir: ' + data_dir +
				'. data_dir should point to the parent directory of .amat files.')

	# We might want to normalize training set after augmentation
	if normalize:
		# Normalize the data with training set statistics
		X_train_mean = np.mean(X_train, axis=0)
		X_train_std = np.std(X_train, axis=0) + 1e-5

		X_train -= X_train_mean
		X_train /= X_train_std

		X_val -= X_train_mean
		X_val /= X_train_std

		X_test -= X_train_mean
		X_test /= X_train_std

		X_train_moments = (X_train_mean, X_train_std)
	return X_train, X_val, X_test, X_train_moments