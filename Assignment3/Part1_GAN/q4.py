import torch
import numpy as np
import pdb
from samplers import distribution3,distribution4
import matplotlib.pyplot as plt


device = torch.device('cpu')
js_model = torch.nn.Sequential(
          torch.nn.Linear(1, 64),
          torch.nn.ReLU(),
          torch.nn.Linear(64, 32),
          torch.nn.ReLU(),
          torch.nn.Linear(32, 1),
          torch.nn.Sigmoid(),
).to(device)
batch_size=512
learning_rate = 1e-4
criterion = torch.nn.BCELoss()
optimizer_js = torch.optim.Adam(js_model.parameters(), lr=learning_rate) 




real_dist = iter(distribution4(batch_size))
fake_dist = iter(distribution3(batch_size))

real_label= torch.full((batch_size,), 1, device=device)
fake_label= torch.full((batch_size,), 0, device=device)

for epoch in range(3000):
        js_model.zero_grad()

        # pdb.set_trace()
        # Train on real
        real_input = torch.from_numpy(next(real_dist).astype(np.float32))
        output_real = js_model(real_input)
        errD_real = criterion(output_real, real_label)
        errD_real.backward()

        # Train on fake
        fake_input = torch.from_numpy(next(fake_dist).astype(np.float32))
        output_fake = js_model(fake_input)
        errD_fake = criterion(output_fake, fake_label)
        errD_fake.backward()

        print(errD_real + errD_fake)
        optimizer_js.step() 


xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
f0 = N(xx)

x_input = torch.from_numpy(xx.astype(np.float32).reshape(1000,1))
d_star = js_model(x_input).squeeze()
f1 = (torch.from_numpy(f0.astype(np.float32)) * d_star / (1-d_star)).detach().numpy()


plt.figure()
plt.plot(xx, f1,label="Estimated distribution")
# plt.plot(xx, N(xx))
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx),label="True distribution")
# plt.show()
plt.legend()
plt.savefig("q4.png")

