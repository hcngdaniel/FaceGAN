#!/usr/bin/env python3
print("Importing Packages...")

import os
import json
from types import SimpleNamespace

import numpy as np
import cv2

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


print("Packages imported")

"""
Load Config File
"""


def to_simple_namespace(d):
    for key, value in filter(lambda x: isinstance(x[1], dict), d.items()):
        d[key] = to_simple_namespace(value)
    return SimpleNamespace(**d)


with open("config.json", "r") as f:
    config = json.load(f)
    config = to_simple_namespace(config)

print("Config file loaded")

"""
Load Parameters
"""
device = torch.device(config.device)

"""
Import and Load Model
"""
exec(f"from model.{config.model.name} import Model")

model = Model()
model.to(device)
if os.path.exists(config.model.path):
    model.load_state_dict(torch.load(config.model.path, map_location=device))
generator = model.generator
critic = model.critic

print("Model loaded")

"""
Load Dataset

Name: CelebA
Shape: (B, 3, 218, 178)

"""

print("Loading dataset...")

train_dataset = torchvision.datasets.CelebA(
    root='./dataset/',
    split='train',
    target_type='identity',
    transform=torchvision.transforms.ToTensor(),
    download=False,
)
valid_dataset = torchvision.datasets.CelebA(
    root='./dataset',
    split='valid',
    target_type='identity',
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

print("Dataset loaded")

"""
Load Dataloader
"""
train_loader = DataLoader(
    train_dataset,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config.data.batch_size,
    shuffle=True,
)

print("Dataloader loaded")

"""
Initialize loss function and optimizer
"""


def grad_penalty(real, fake):

    epsilon = torch.rand(config.data.batch_size, 1, 1, 1)
    epsilon = epsilon.expand(real.size())
    epsilon.to(device)
    interpolates = torch.mul(epsilon, real) + torch.mul(torch.ones(real.size()) - epsilon, fake)
    interpolates.to(device)

    critic_interpolates = critic(interpolates)

    grads = torch.autograd.grad(
        inputs=interpolates,
        outputs=critic_interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_penalties = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    grad_penalties = config.train.Lambda * grad_penalties

    return grad_penalties


optimizer = torch.optim.RMSprop(model.parameters(), lr=config.train.lr)


"""
Training
"""

for epoch in range(config.train.start_epoch, config.train.end_epoch):
    for batch, (real, _) in enumerate(train_loader):

        print(f"epoch {epoch} batch {batch}")

        for p in critic.parameters(): p.requires_grad = True

        for iter_c in range(config.train.critic_iters):
            real = real.to(device)

            real_e = critic(real)
            real_l = real_e.mean()

            noise = torch.randn(config.data.batch_size, 100, device=device)
            fake = generator(noise)

            fake_e = critic(fake)
            fake_l = fake_e.mean()

            gp = grad_penalty(real, fake)

            c_loss = fake_l - real_l + gp

            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

        for p in critic.parameters(): p.requires_grad = False

        noise = torch.randn(config.data.batch_size, 100, device=device)
        fake = generator(noise)
        g_loss = -fake.mean()

        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), config.model.path)
        print("saved")

        noise = torch.randn(1, 100, device=device)
        fake_sample = generator(noise).detach().cpu().numpy()[0]
        fake_cv = cv2.cvtColor(fake_sample, cv2.COLOR_RGB2BGR)
        cv2.imshow("sample")
        cv2.waitKey(1)
