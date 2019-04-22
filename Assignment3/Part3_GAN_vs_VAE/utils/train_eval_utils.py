# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, optimizer, loader_train, loader_val, num_epochs, device):
    """
    Trains model.
    
    Inputs:
    - model: Module object which is our model
    - loader_train: Training set loader.
    - loader_val: Validation set loader.
    - optimizer: Optimizer object for training the model
    - num_epochs: Number of epochs to train
    
    Returns:
    - train_hist: History of training
    """
    model = model.to(device=device)  # Put model parameters on GPU or CPU
    train_history = {'train_elbo_hist':[], 'val_elbo_hist':[],
    'train_likelihood_hist':[], 'val_likelihood_hist':[],
    'train_kl_hist':[], 'val_kl_hist':[]}
    
    for epoch in range(num_epochs):
        total_train_size = 0
        total_train_elbo = 0
        total_train_likelihood = 0
        total_train_kl = 0
        for it, X in enumerate(loader_train):
            # Move inputs to specified device
            X = X[0].to(device=device, dtype=torch.float32)

            # Forward pass
            model.train()
            x_hat, z, mean, logvar = model(X)
            elbo, likelihood, kl = criterion(x_hat, X, mean, logvar)
            negative_elbo = -elbo

            # Backward pass
            optimizer.zero_grad() # Zero out the gradients
            negative_elbo.backward() # Compute gradient of negative elbo w.r.t. model parameters
            optimizer.step() # Make a gradient update step
            minibatch_size = X.size(0)

            # Epoch loss and accuracy average
            total_train_size += minibatch_size
            total_train_elbo += (minibatch_size * elbo).item()
            total_train_likelihood += (minibatch_size * likelihood).item()
            total_train_kl += (minibatch_size * kl).item()

        ### Update train_history
        train_epoch_elbo = total_train_elbo / total_train_size
        train_epoch_likelihood = total_train_likelihood / total_train_size
        train_epoch_kl = total_train_kl / total_train_size

        train_history['train_elbo_hist'].append(train_epoch_elbo)
        train_history['train_likelihood_hist'].append(train_epoch_likelihood)
        train_history['train_kl_hist'].append(train_epoch_kl)
        # val_loss_hist & val_acc_hist
        val_elbo, val_likelihood, val_kl = evaluation(model, loader_val, criterion, device=device)
        train_history['val_elbo_hist'].append(val_elbo)
        train_history['val_likelihood_hist'].append(val_likelihood)
        train_history['val_kl_hist'].append(val_kl)

        # Print training process
        print('Epoch %d:' % (epoch+1))
        print('Train: elbo %.4f, likelihood %.4f, KL %.4f' % (train_epoch_elbo, train_epoch_likelihood, train_epoch_kl))
        print('Validation: elbo %.4f, likelihood %.4f, KL %.4f' % (val_elbo, val_likelihood, val_kl))
        print('-----------')
    return train_history


def evaluation(model, loader, criterion, device):
    """
    Returns the elbo, likelihood & kl of the model on loader data.
    
    Inputs:
    model: Module object which is our model
    loader: DataLoader object
    
    Returns:
    - elbo
    - likelihood
    - kl
    """
    total_elbo = 0
    total_likelihood = 0
    total_kl = 0
    model.eval()  # change model mode to eval
    with torch.no_grad():  # temporarily set all requires_grad flags to False
        for X in loader:
            # Move inputs to specified device
            X = X[0].to(device=device, dtype=torch.float32)
            
            # Compute elbo, likelihood & kl
            x_hat, z, mean, logvar = model(X)
            elbo, likelihood, kl = criterion(x_hat, X, mean, logvar)
            total_elbo += elbo
            total_likelihood += likelihood
            total_kl += kl

    total_elbo /= len(loader)
    total_likelihood /= len(loader)
    total_kl /= len(loader)
    return total_elbo, total_likelihood, total_kl


def criterion(X_hat, X, mean, logvar):
    minibatch_size = X.shape[0]
    likelihood = - (1 / minibatch_size) * F.mse_loss(X_hat, X, reduction='sum')
    kl = (1 / minibatch_size) * 0.5 * torch.sum(- 1 - logvar + torch.pow(mean, 2) + torch.exp(logvar))
    elbo = likelihood - kl
    return elbo, likelihood, kl