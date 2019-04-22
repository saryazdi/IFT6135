# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

def importance_sampling(model, loader, device, K=200, num_latent=100):
    """
    Returns the log-likelihood estimate of the model on loader data.
    
    Inputs:
    model: Module object which is our model
    loader: DataLoader object
    device: Device to use
    
    Returns:
    - log-likelihood estimate of the model on loader data
    """
    D = 784
    model.eval()  # change model mode to eval
    logp_total = 0
    num_total = 0
    with torch.no_grad():  # temporarily set all requires_grad flags to False
        for X in loader:
            # Move inputs to specified device
            X = X[0].to(device=device, dtype=torch.float32)
            M = X.shape[0]
            X = X.view(M, D)
            Z = torch.randn(M, K, num_latent) # Z gets "reparameterized" in minibatch_importance_sampling() so that: z~q(z|x)
            Z = Z.to(device=device, dtype=torch.float32)

            # Compute logp
            logp = minibatch_importance_sampling(model, X, Z, device)
            logp_total += torch.sum(logp)
            num_total += M
    return logp_total / num_total


def minibatch_importance_sampling(model, X, Z, device):
    '''
    Inputs:
    - model: Trained VAE
    - X: An (M, D) tensor for x_i's
    - Z: An (M, K, L) tensor of z_ik's
    
    Returns:
    - (logp(x_1), ..., logp(x_M)) estimates of size (M,)
    '''
    M, D = X.shape
    M, K, L = Z.shape
    X_img = X.view(M, 1, 28, 28)
    
    log_probabilities = torch.zeros((M, K)).to(device)
    
    model.eval()  # change model mode to eval
    with torch.no_grad():
        for k in range(K):
            z = Z[:, k, :]
            mean, logvar = model.encode(X_img)
            z = mean + (z * (torch.exp(logvar * 0.5))) # Reparameterize z
            X_hat_probs = model.decode(z, apply_sigmoid=True).view(M, D)

            log_p_posterior = torch.sum((X * torch.log(X_hat_probs+1e-7)) + ((1-X) * torch.log(1 - X_hat_probs+1e-7)), dim=1)
            log_p_prior = torch.sum(Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z), dim=1)
            log_q_posterior = torch.sum(Normal(mean, torch.exp(0.5 * logvar)).log_prob(z), dim=1) # Normal requires std NOT var
            log_probabilities[:,k] = log_p_posterior + log_p_prior - log_q_posterior
        
    logp = torch.logsumexp(log_probabilities, dim=1) - np.log(K)
    return logp