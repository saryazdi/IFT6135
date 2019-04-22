import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self, num_latent):
		super(Generator, self).__init__()
		self.num_latent = num_latent

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
		nn.Tanh())

	
	def sample(self, eps):
		return self.decode(eps)
	
	def decode(self, z):
		logits = self.decoder(z.view(z.shape[0], -1, 1, 1))
		return logits

	def forward(self, z):
		return self.decode(z)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.discrim = nn.Sequential(
		nn.Conv2d(3, 128, 4, stride=2, padding=1),
		nn.BatchNorm2d(128),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(128, 256, 4, stride=2, padding=1),
		nn.BatchNorm2d(256),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(256, 512, 4, stride=2, padding=1),
		nn.BatchNorm2d(512),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(512, 1, 4, stride=1, padding=0)
		)


	def forward(self, x):
		return self.discrim(x)