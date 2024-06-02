import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from process import *


class GansDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx]}


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_gan(dataloader, generator, discriminator, output_size, device, epochs=50):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)

    real_label = 1
    fake_label = 0

    # 准备收集生成的数据
    all_generated_data = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            # 更新判别器
            discriminator.zero_grad()
            real_cpu = data['data'].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, output_size, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            # 更新生成器
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            # 收集生成的数据
            all_generated_data.append(fake.detach().cpu())

            if i % 100 == 0:
                print(
                    f'[{epoch}/{epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {round(D_G_z1,3)}/{round(D_G_z2,3)}')

    print('Finished Training')

    # 将所有生成的数据连接成一个大数组
    all_generated_data = torch.cat(all_generated_data, dim=0)

    # 如果生成的数据数量超过了原始数据的数量，我们截取前N个
    all_generated_data = all_generated_data[:len(dataloader.dataset)]

    return all_generated_data.numpy()


# 后续的部分保持不变


if __name__ == '__main__':
    data = pd.read_csv("./data/sh600000.csv")
    data = process_data(data)
    data.drop(['date', 'ticker', 'qfq_factor'], axis=1, inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = data.iloc[:, :-1].values

    gandataset = GansDataset(X_train)
    gandataloader = DataLoader(gandataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]
    generator = Generator(input_size, input_size)
    discriminator = Discriminator(input_size)

    generated_features = train_gan(gandataloader, generator, discriminator, input_size, device, epochs=20)

    print(generated_features.shape)  # Check the dimensions of generated data
    print(generated_features)
