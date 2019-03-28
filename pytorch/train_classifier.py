import torch
from torch import nn, optim
from torchvision import datasets, transforms
import numpy as np

EPOCHS = 50
ETA = 1e-4


def get_MNIST(train=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    if train:
        dataset = datasets.MNIST(
            "./data",
            train=train,
            transform=transform,
            download=True
        )

        n = len(dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)

        train_sampler = torch.utils.data.SubsetRandomSampler(indices[:int(n*0.8)])
        valid_sampler = torch.utils.data.SubsetRandomSampler(indices[int(n*0.8):])

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            sampler=train_sampler
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            sampler=valid_sampler
        )

        return train_loader, valid_loader

    else:
        dataset = datasets.MNIST(
            "./data",
            train=train,
            transform=transform,
            download=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=True
        )

        return test_loader


def train_classifier():
    from Classifier import Classifier

    device = torch.device("cuda")
    clf = Classifier().to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(clf.parameters(), lr=ETA)

    train_loader, valid_loader = get_MNIST(True)

    for e in range(EPOCHS):
        train_loss = 0.0

        for images, labels, in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logps = clf(images)
            loss = criterion(logps, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            clf.eval()

            valid_loss = 0.0
            valid_acc = 0.0

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                # get loss on valid set
                logps = clf(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()

                # calculate accuracy on valid set.
                ps = torch.exp(logps)
                cls_ps, topk = ps.topk(1, dim=1)
                equal = topk == labels.view(*topk.size())
                valid_acc += torch.mean(equal.float())

            clf.train()

            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Valid loss: {valid_loss:.8f}")
            print(f"Valid acc: {valid_acc:.4f}")

    clf.save("ckpts/clf.pth")


if __name__ == "__main__":
    train_classifier()
    # test_clfoder()
