import torch
from torch import nn
from StackedAutoEncoder import StackedAutoEncoder

CKPT = "ckpts/sae.pth"


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.sae = StackedAutoEncoder()
        self.sae.load(CKPT)

        for param in self.sae.parameters():
            param.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Linear(8*8*32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        n = x.size(0)

        x, _ = self.sae(x, index=1)
        x = x.view(n, -1)

        x = self.classifier(x)

        return x

    def save(self, ckpt):
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt)
        print("Classifier was saved.")

    def load(self, ckpt):
        state_dict = torch.load(ckpt)
        self.load_state_dict(state_dict)
        print("Classifier was loaded.")
