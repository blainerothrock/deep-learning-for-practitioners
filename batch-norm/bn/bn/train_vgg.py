import torch
import torchvision
import torchvision.transforms as transforms

from bn.VGG11 import VGG11


def train_vgg(batch_size=4, batch_norm=False, noise=False, learning_rate=0.01, num_epochs=2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = VGG11(num_classes=10, init_weights=True, batch_norm=batch_norm, noise_injection=noise)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0

        print(trainloader)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # print(inputs.shape)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3d' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
