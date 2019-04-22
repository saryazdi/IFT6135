# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/04/10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
def calc_gradient_penalty(discriminator, real_data, fake_data, batch_size, use_cuda):
    # pdb.set_trace()
    lambdaa=10.
    #print real_data.size()
    alpha = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    alpha = alpha.expand(batch_size, real_data.size(1), real_data.size(2), real_data.size(3))

    alpha = alpha.cuda(0) if use_cuda else alpha 
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda(0) if use_cuda else interpolates
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(0) if use_cuda else  torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambdaa
    return gradient_penalty


def train_model(discriminator, generator, optimizer_d, optimizer_g,
                 loader_train, loader_val, num_iterations, d_iterations,
                 device, use_cuda, batch_size, scale):
    """
    Trains model.
    
    Inputs:
    - model: Module object which is our model
    - loader_train: Training set loader.
    - loader_val: Validation set loader.
    - optimizer: Optimizer object for training the model
    - num_iterations: Number of epochs to train
    
    Returns:
    - train_hist: History of training
    """

    discriminator = discriminator.to(device=device)  # Put discriminator parameters on GPU or CPU
    generator = generator.to(device=device)  # Put generator parameters on GPU or CPU

    one = torch.FloatTensor([1]).to(device=device)
    minus_one = torch.FloatTensor([-1]).to(device=device)

    loader_train_it = iter(loader_train)
    
    for it in range(num_iterations):
    
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        ########################################################################################
        for d_it in range(d_iterations):
            discriminator.zero_grad()
            # from utils.plotter import plot_and_save_images
            
            # Train on real
            try:
                data = next(loader_train_it)[0]
                # plot_and_save_images(np.transpose(data.numpy()[:10] * 255., (0, 2, 3, 1)).astype('uint8'), 'GAN')            
            except StopIteration:
                loader_train_it = iter(loader_train)
                data = next(loader_train_it)[0]
            if (data.size()[0] != batch_size):
                continue
            real_input = data.to(device=device, dtype=torch.float32)
            output_real = discriminator(real_input)
            loss_real = output_real.mean()
            loss_real.backward(minus_one)


            # Train on fake
            noise = Variable(torch.randn(batch_size, 100).to(device=device))
            fake_input = generator(noise)
            output_fake = discriminator(fake_input.data)
            loss_fake = output_fake.mean()
            loss_fake.backward(one)


            # Train on gradient penalty
            gradient_penalty = calc_gradient_penalty(discriminator, real_input.data, fake_input.data,
                                                    batch_size, use_cuda)
            gradient_penalty.backward(retain_graph=True)

            optimizer_d.step() 

        ########################################################################################
        
        for p in discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        generator.zero_grad()

        noise = Variable(torch.randn(batch_size, 100).to(device=device))
        fake_input = generator(noise)
        score_generator = discriminator(fake_input)
        loss_gen = score_generator.mean()
        loss_gen.backward(minus_one)
        optimizer_g.step()


        print("Iteration: {}, Loss generator: {}, Loss discriminator {}".format(it,loss_gen.cpu().data.numpy().reshape(1,)[0],loss_real.cpu().data.numpy().reshape(1,)[0]-loss_fake.cpu().data.numpy().reshape(1,)[0]))


        if (it) % 100 == 0:
            eps = torch.randn(10, 100).to(device)
            generator.eval()
            with torch.no_grad():
                gan_generations = generator.sample(eps).cpu().numpy()
            gan_generations = np.transpose(scale(gan_generations), (0, 2, 3, 1)).astype('uint8')
            images=gan_generations

            fig = plt.figure(figsize=(8, 8))
            fig.suptitle('GAN SVHN generations')
            for i in range(len(images)):
                plt.subplot(5, 5, i+1)
                plt.imshow(images[i])
                plt.axis('off')
            plt.savefig('figures/GAN_SVHN_generations.png')
