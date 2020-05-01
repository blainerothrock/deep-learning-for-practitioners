from enum import Enum
import importlib

import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

import seaborn as sns
from tensorboardX import SummaryWriter


class ModelType(str, Enum):
    SIMPLE_FF = 'SimpleFF'
    SIMPLE_FF_BN = 'SimpleFFBN'
    SIMPLE_FF_BN_NOISE = 'SimpleFFBNNoise'


def train(model_type=ModelType.SIMPLE_FF, batch_size=128, num_epochs=2):
    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    module = importlib.import_module("bn")
    class_ = getattr(module, model_type.value)
    model = class_()

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)

    loss_arr = []
    l1_weights = []
    l2_weights = []

    writer = SummaryWriter(log_dir='runs/' + model_type.value)

    for epoch in range(num_epochs):

        for i, data in enumerate(trainloader, 0):

            n_iter = (epoch * len(trainloader)) + i

            inputs, labels = data

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_arr.append(loss.item())

            writer.add_scalar('loss', loss.item(), n_iter)
            writer.add_scalar('inputs/layer1/mean', model.l1_inp.mean(), n_iter)
            writer.add_scalar('inputs/layer2/mean', model.l2_inp.mean(), n_iter)

            if i % 10 == 0:
                inputs = inputs.view(inputs.size(0), -1)

                model.eval()
                print('loss: %0.2f' % loss.item())

                model.train()

    # compute summary
    l1_mean = [x[0] for x in model.l1_dist]
    l1_std = [x[1] for x in model.l1_dist]
    l2_mean = [x[0] for x in model.l2_dist]
    l2_std = [x[1] for x in model.l2_dist]

    return l1_mean, l1_std, l2_mean, l2_std, loss_arr, model_type.value
