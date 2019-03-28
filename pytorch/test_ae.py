import torch
from torch import nn, optim
from torchvision import datasets, transforms


EPOCHS = 10
ETA = 1e-2


def get_cifar10():
    dataset = datasets.CIFAR10(
        "./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    return train_loader


def train_encoder():
    from AutoEncoder import AutoEncoder

    device = torch.device("cuda")
    enc = AutoEncoder(3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(enc.parameters(), lr=ETA)

    loader = get_cifar10()

    for e in range(EPOCHS):
        train_loss = 0.0

        for images, _, in loader:
            images = images.to(device)

            _, decoded = enc(images)
            assert(decoded.size() == images.size())
            loss = criterion(decoded, images)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Loss: {train_loss:.8f}")

    enc.save("ckpts/encoder_test.pth")


def test_encoder():
    from AutoEncoder import AutoEncoder
    import matplotlib.pyplot as plt

    device = torch.device("cuda")

    enc = AutoEncoder(3).to(device)
    enc.load("ckpts/encoder_test.pth")

    loader = iter(get_cifar10())

    images, _ = next(loader)

    with torch.no_grad():
        _, decoded = enc(images.to(device))
        decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()

        for i in range(10):
            plt.imshow(images[i].permute(1, 2, 0).numpy())
            plt.show()

            plt.imshow(decoded[i])
            plt.show()


if __name__ == "__main__":
    # train_encoder()
    test_encoder()
