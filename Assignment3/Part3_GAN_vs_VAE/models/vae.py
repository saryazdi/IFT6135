# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import torch
import torch.nn as nn

# Create model
class VAE(nn.Module):
	def __init__(self, num_latent):
		super(VAE, self).__init__()
		self.num_latent = num_latent
		
		# build encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
			nn.BatchNorm2d(num_features=32),
			nn.ELU(),
			nn.AvgPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
			nn.BatchNorm2d(num_features=64),
			nn.ELU(),
			nn.AvgPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=256, kernel_size=6),
			nn.BatchNorm2d(num_features=256),
			nn.ELU()
		)
		self.encoder_fc = nn.Linear(256, num_latent * 2)
		
		# build decoder
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(100, 1024, 4, stride=1, padding=0),
	        nn.BatchNorm2d(num_features=1024),
	        nn.ReLU(inplace=True),
	        nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
	        nn.BatchNorm2d(num_features=512),
	        nn.ReLU(inplace=True),
	        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
	        nn.BatchNorm2d(num_features=256),
	        nn.ReLU(inplace=True),
	        nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
		)
	
	def sample(self, eps):
		return self.decode(eps, apply_tanh=True)
	
	def encode(self, x):
		mean_logvar = self.encoder_fc(self.encoder(x).view(x.shape[0], -1))
		mean, logvar = mean_logvar[:, :self.num_latent], mean_logvar[:, self.num_latent:]
		return mean, logvar
	
	def reparameterize(self, mean, logvar):
		eps = torch.randn_like(mean)
		return mean + (eps * (torch.exp(logvar * 0.5)))

	def decode(self, z, apply_tanh=True):
		logits = self.decoder(z.view(z.shape[0], -1, 1, 1))
		if apply_tanh:
			probs = torch.tanh(logits)
			return probs
		return logits

	def forward(self, x):
		mean, logvar = self.encode(x)
		z = self.reparameterize(mean, logvar)
		X_hat_params = self.decode(z, apply_tanh=True)
		return X_hat_params, z, mean, logvar