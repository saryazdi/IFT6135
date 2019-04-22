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
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5),
            nn.ELU()
        )
        self.encoder_fc = nn.Linear(256, num_latent * 2)
        
        # build decoder
        self.decoder_fc = nn.Linear(num_latent, 256)
        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2)
        )
    
    def encode(self, x):
        mean_logvar = self.encoder_fc(self.encoder(x).view(x.shape[0], -1))
        mean, logvar = mean_logvar[:, :self.num_latent], mean_logvar[:, self.num_latent:]
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return mean + (eps * (torch.exp(logvar * 0.5)))

    def decode(self, z, apply_sigmoid=True):
        logits = self.decoder(self.decoder_fc(z).view(z.shape[0], -1, 1, 1))
        if apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        X_hat = self.decode(z, apply_sigmoid=True)
        return X_hat, z, mean, logvar