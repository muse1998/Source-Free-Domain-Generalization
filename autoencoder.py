import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    It is an autoencoder consist of linear layers.
    We use cosine loss to train it, so we call it CAE in the paper.
    """
    def __init__(self, in_shape=512):
        super(AutoEncoder, self).__init__()
        print(type(in_shape))
        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256))
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True), 
            nn.Linear(512, in_shape))
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x / x.norm(dim=1, keepdim=True)
        return x

def auto_encoder(in_shape=512):
    return AutoEncoder(in_shape)