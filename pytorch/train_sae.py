import torch
from torch import nn, optim
from torchvision import datasets, transforms


EPOCHS = 20
ETA = 3e-3


def get_MNIST():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(
        "./data",
        train=True,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    return train_loader


def train_encoder():
    from StackedAutoEncoder import StackedAutoEncoder

    device = torch.device("cuda")
    enc = StackedAutoEncoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(enc.parameters(), lr=ETA)

    loader = get_MNIST()

    # train 2 encoders
    # 두 개의 인코더를 학습
    for i in range(2):
        print(f"Encoder {i+1}/{2}")
        for e in range(EPOCHS):
            train_loss = 0.0

            for images, _, in loader:
                images = images.to(device)

                # get encoded image from previous encoder.
                # 이전 인코더로부터 인코딩된 이미지 얻어옴.
                with torch.no_grad():
                    enc.eval()
                    if i != 0:
                        target, _ = enc(images, index=i-1)
                    else:
                        target = images
                    enc.train()

                # train current encoder.
                # 현재 인코더 학습
                _, decoded = enc(images, index=i)
                loss = criterion(decoded, target)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(loader)
            print(f"Epochs {e+1}/{EPOCHS}")
            print(f"Loss: {train_loss:.8f}")

    enc.save("ckpts/sae.pth")


if __name__ == "__main__":
    train_encoder()
    # test_encoder()
