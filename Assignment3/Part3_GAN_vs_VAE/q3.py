import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DataPreparation.dataset_preparation import get_SVHN_dataset
from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

data_dir = 'Dataset/SVHN/'
validation_split = 0.2
seed = 6135

X_train, y_train, X_val, y_val, X_test, y_test, X_train_max = get_SVHN_dataset(data_dir, validation_split, seed, normalize=True)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_val shape: ', X_test.shape)
print('y_val shape: ', y_test.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


X_train_ = TensorDataset(torch.from_numpy(X_train))
loader_train  = DataLoader(X_train_, batch_size=64, shuffle=True)

X_val_ = TensorDataset(torch.from_numpy(X_val))
loader_val = DataLoader(X_val_, batch_size=64, shuffle=False)

X_test_ = TensorDataset(torch.from_numpy(X_test))
loader_test = DataLoader(X_test_, batch_size=64, shuffle=False)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Using device=GPU') if use_cuda else print('Using device=CPU')


from models.gan import Generator, Discriminator
num_latent = 100
generator = Generator(num_latent).to(device)
discriminator = Discriminator().to(device)

# Hyperparameters
learning_rate = 1e-4
num_iterations = 25
d_iterations = 5

from utils.train_eval_utils_gan import train_model
print('~~~ Training with GPU ~~~') if use_cuda else print('~~~ Training with CPU ~~~\n')
num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print('Generator has %.2fM trainable parameters.\n' % (num_params/1e6))
num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
print('Discriminator has %.2fM trainable parameters.\n' % (num_params/1e6))

optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
train_model(discriminator, generator, optimizer_d, optimizer_g,
            loader_train, loader_val, num_iterations, d_iterations, device,use_cuda)



