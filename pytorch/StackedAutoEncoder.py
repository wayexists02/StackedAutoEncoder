import torch
from torch import nn
from AutoEncoder import AutoEncoder


class StackedAutoEncoder(nn.Module):

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        # stacking 2 auto encoders
        # 2개의 오토인코더를 스택
        self.encoder1 = AutoEncoder(1)
        self.encoder2 = AutoEncoder(32)

        self.encoders = [
            self.encoder1,
            self.encoder2,
        ]

    def forward(self, x, index):
        encoded = x
        for i in range(index+1):
            encoded, decoded = self.encoders[i](encoded)

        return encoded, decoded

    def save(self, ckpt):
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt)
        print("AutoEncoder was saved.")

    def load(self, ckpt):
        state_dict = torch.load(ckpt)
        self.load_state_dict(state_dict)
        print("AutoEncoder was loaded.")
