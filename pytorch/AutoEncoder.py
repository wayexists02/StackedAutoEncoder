import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, in_channels):
        super(AutoEncoder, self).__init__()

        self.encoder_layers = self._encoder_net(in_channels)
        self.decoder_layers = self._decoder_net(in_channels)

    def forward(self, x):
        encoded, indices = self._encode(x)
        decoded = self._decode(encoded, indices)

        return encoded, decoded

    def save(self, ckpt):
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt)
        print("AutoEncoder was saved.")

    def load(self, ckpt):
        state_dict = torch.load(ckpt)
        self.load_state_dict(state_dict)
        print("AutoEncoder was loaded.")

    def _encode(self, x):
        """
        Propagate through encoder part.
        인코더 부분 통과
        """

        indices_list = []
        for layer in self.encoder_layers:
            if type(layer) is nn.MaxPool2d:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        return x, indices_list

    def _decode(self, x, indices_list):
        """
        propagate through decoder part.
        디코더 부분 통과
        """

        ind = len(indices_list) - 1
        for layer in self.decoder_layers:
            if type(layer) is nn.MaxUnpool2d:
                x = layer(x, indices_list[ind])
                ind -= 1
            else:
                x = layer(x)

        return x

    def _encoder_net(self, in_channels):
        """
        Build encoder part in network.
        네트워크에서 인코더 파트를 빌드

        Arguments:
        ----------
        :in_channels num of channels of input of this net.

        Returns:
        --------
        :layers encoder layer list
        """
        
        self.encoder_layer1 = nn.Conv2d(in_channels, 32, (3, 3), stride=1, padding=1)
        self.encoder_layer2 = nn.Sigmoid()

        self.encoder_layer3 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)
        self.encoder_layer4 = nn.Sigmoid()

        self.encoder_layer5 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)
        self.encoder_layer6 = nn.Sigmoid()

        self.encoder_layer7 = nn.MaxPool2d((2, 2), stride=2, padding=0,
                                           return_indices=True)

        layers = [
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
            self.encoder_layer5,
            self.encoder_layer6,
            self.encoder_layer7,
        ]

        return layers

    def _decoder_net(self, in_channels):
        """
        Build decoder part of this net.
        디코더 부분 빌드

        Arguments:
        ----------
        :in_channels num of output channels (same as num of input channels)

        Returns:
        --------
        :layers decoder layers list
        """
        
        self.decoder_layer1 = nn.MaxUnpool2d((2, 2), stride=2, padding=0)

        self.decoder_layer2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=1, padding=1)
        self.decoder_layer3 = nn.Sigmoid()

        self.decoder_layer4 = nn.ConvTranspose2d(32, 32, (3, 3), stride=1, padding=1)
        self.decoder_layer5 = nn.Sigmoid()

        self.decoder_layer6 = nn.ConvTranspose2d(32, in_channels, (3, 3), stride=1, padding=1)
        self.decoder_layer7 = nn.Sigmoid()

        layers = [
            self.decoder_layer1,
            self.decoder_layer2,
            self.decoder_layer3,
            self.decoder_layer4,
            self.decoder_layer5,
            self.decoder_layer6,
            self.decoder_layer7,
        ]

        return layers
