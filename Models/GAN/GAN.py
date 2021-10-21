import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self,inp_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )


    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,z,inp_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, inp_dim),
            nn.Tanh(), 
        )


    def forward(self,x):
        return self.gen(x)
#GANs are very sensitive to hyperparameters.So be careful with these values.
device="cuda"
lr=3e-4
z=64
inp_dim=28*28*1 # 784 for MNIST
batch_size=32
num_epochs=50

D=Discriminator(inp_dim).to(device)
G=Generator(z,inp_dim).to(device)
const_noise=torch.randn((batch_size,z)).to(device)
transforms = tr.Compose(
    [tr.ToTensor(), tr.Normalize((0.5,), (0.5,)),]
)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0



dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_D = optim.Adam(D.parameters(), lr=lr)
opt_G = optim.Adam(G.parameters(), lr=lr)
Loss_Type = nn.BCELoss()


for epoch in range(num_epochs):
    for idx,(real,labels) in enumerate(data_loader):
        real=real.view(-1,784).to(device)
        batch_size=real.shape[0]
        
        noise = torch.randn(batch_size, z).to(device)
        fake = G(noise)
        disc_real = D(real).view(-1)
        lossD_real = Loss_Type(disc_real, torch.ones_like(disc_real))
        disc_fake = D(fake).view(-1)
        lossD_fake = Loss_Type(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        D.zero_grad()
        lossD.backward(retain_graph=True)
        opt_D.step()

        output = D(fake).view(-1)
        lossG = Loss_Type(output, torch.ones_like(output))
        G.zero_grad()
        lossG.backward()
        opt_G.step()


        if idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(data_loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = G(const_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1