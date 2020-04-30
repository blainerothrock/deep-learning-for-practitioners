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


class ModelType(str, Enum):
    FF = 'Model'
    BN = 'ModelBN'


def train(model_type=ModelType.FF, batch_size=128, num_epochs=2):
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

    for epoch in range(num_epochs):

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            loss_arr.append(loss.item())

            if i % 10 == 0:
                inputs = inputs.view(inputs.size(0), -1)

                model.eval()
                print('loss: %0.2f' % loss.item())

                model.train()

    l1_mean = [x[0] for x in model.l1_dist]
    plt.plot(l1_mean, 'r', label='%s layer 1 input mean' % model_type.value)
    plt.savefig('imgs/%s_l1_mean.png' % model_type.value)
    plt.close()

    l1_std = [x[1] for x in model.l1_dist]
    plt.plot(l1_std, 'r', label='%s layer 1 input std' % model_type.value)
    plt.savefig('imgs/%s_l1_std.png' % model_type.value)
    plt.close()

    l2_mean = [x[0] for x in model.l2_dist]
    plt.plot(l2_mean, 'r', label='%s layer 2 input mean' % model_type.value)
    plt.savefig('imgs/%s_l2_mean.png' % model_type.value)
    plt.close()

    l2_std = [x[1] for x in model.l2_dist]
    plt.plot(l2_std, 'r', label='%s layer 2 input std' % model_type.value)
    plt.savefig('imgs/%s_l2_std.png' % model_type.value)
    plt.close()

    plt.plot(loss_arr, 'r', label='%s loss' % model_type.value)
    plt.savefig('imgs/%s_loss.png' % model_type.value)
    plt.close()
