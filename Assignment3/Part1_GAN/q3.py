import torch
import numpy as np
import pdb
import torch.autograd as autograd
import matplotlib.pyplot as plt

def q3_js(phi=0.):
    device = torch.device('cpu')
    js_model = torch.nn.Sequential(
              torch.nn.Linear(2, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 32),
              torch.nn.ReLU(),
              torch.nn.Linear(32, 1),
              torch.nn.Sigmoid(),
    ).to(device)
    batch_size=512
    learning_rate = 1e-3
    criterion = torch.nn.BCELoss()
    optimizer_js = torch.optim.Adam(js_model.parameters(), lr=learning_rate) 




    real_data= torch.full((batch_size,1), 0)
    fake_data= torch.full((batch_size,1), phi)

    real_label= torch.full((batch_size,), 1, device=device)
    fake_label= torch.full((batch_size,), 0, device=device)

    for epoch in range(500):
            js_model.zero_grad()
            z = torch.FloatTensor(batch_size,1).uniform_(0, 1)

            # Train on real
            real_input = torch.cat((real_data,z),1)   
            output_real = js_model(real_input)
            errD_real = criterion(output_real, real_label)
            errD_real.backward()

            # Train on fake
            fake_input = torch.cat((fake_data,z),1)
            output_fake = js_model(fake_input)
            errD_fake = criterion(output_fake, fake_label)
            errD_fake.backward()

            js = torch.log(output_real).mean() /2 + torch.log(1-output_fake).mean() /2 + np.log(2) 
            # print(js)
            optimizer_js.step() 
    return js.detach().numpy()


def q3_wd(phi=0.):

    def calc_gradient_penalty(netD, real_data, fake_data,batch_size):
        # pdb.set_trace()
        lambdaa=10.
        #print real_data.size()
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())


        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambdaa
        return gradient_penalty


    device = torch.device('cpu')
    wd_model = torch.nn.Sequential(
              torch.nn.Linear(2, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 1),
              # torch.nn.Sigmoid(),
    ).to(device)
    batch_size=512
    learning_rate = 5e-3
    optimizer_wd = torch.optim.Adam(wd_model.parameters(), lr=learning_rate) 




    real_data= torch.full((batch_size,1), 0)
    fake_data= torch.full((batch_size,1), phi)


    one = torch.FloatTensor([1])
    mone = one * -1
    for epoch in range(1000):
            wd_model.zero_grad()
            z = torch.FloatTensor(batch_size,1).uniform_(0, 1)

            # Train on real
            real_input = torch.cat((real_data,z),1)   
            output_real = wd_model(real_input)
            errD_real = output_real.mean()
            errD_real.backward(mone)

            # Train on fake
            fake_input = torch.cat((fake_data,z),1)
            output_fake = wd_model(fake_input)
            errD_fake = output_fake.mean()
            errD_fake.backward(one)

            # Train on gradient penalty
            gradient_penalty = calc_gradient_penalty(wd_model, real_input.data, fake_input.data,batch_size)
            gradient_penalty.backward()

            wd = errD_real - errD_fake
            # print(wd)

            optimizer_wd.step() 

    return wd.detach().numpy()

phis=np.linspace(-1.,1.,21)
# phis=[-1.,0,1.]
all_js=[]
all_wd=[]
for phi in phis:
    all_js.append(q3_js(phi=phi)) # Approximate the JS divergence
    all_wd.append(q3_wd(phi=phi)) # Approximate the Wasserstein distance

# pdb.set_trace()



plt.figure()
plt.plot(phis,all_js,'ro',label="Jenson-Shannon Estimate")
plt.legend()
# plt.show()
plt.savefig("q3js.png")

plt.figure()
plt.plot(phis,all_wd,'b^',label="Wasserstein Distance Estimate")
plt.legend()
# plt.show()
plt.savefig("q3wd.png")