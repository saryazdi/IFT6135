import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


def plot_and_save_images(images, modelname):
	fig = plt.figure(figsize=(10, 10))
	fig.suptitle(f'{modelname} SVHN generations')
	for i in range(len(images)):
		plt.subplot(8, 8, i+1)
		plt.imshow(images[i])
		plt.axis('off')
	plt.savefig(f'figures/{modelname}_SVHN_generations.png')
	plt.show()


def plot_and_save_disentanglement(images, num_rows, num_cols, modelname):
	images = np.reshape(images, (-1, 32, 32, 3))
	fig = plt.figure(figsize=(8, 20))
	fig.suptitle(f'{modelname} SVHN disentanglement')
	for i in range(len(images)):
		plt.subplot(num_rows, num_cols, i+1)
		plt.imshow(images[i])
		plt.axis('off')
	plt.savefig(f'figures/{modelname}_SVHN_disentanglement.png')
	plt.show()


def plot_and_save_interpolation(images, num_cols, spacename, modelname):
	images = np.reshape(images, (-1, 32, 32, 3))
	fig = plt.figure(figsize=(20, 4))
	fig.suptitle(f'{modelname} SVHN {spacename} interpolation')
	for i in range(len(images)):
		plt.subplot(1, num_cols, i+1)
		plt.imshow(images[i])
		plt.axis('off')
	plt.savefig(f'figures/{modelname}_SVHN_{spacename}_interpolation.png')
	plt.show()

def sample_saver(images, path):
	for i, image in enumerate(images):
		scipy.misc.toimage(image, cmin=0., cmax=255.).save(f'{path}\\{i+1}.jpg')
