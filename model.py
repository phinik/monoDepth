import torch
import torch.nn as nn

from os.path import join
from fast_depth import FastDepth
from best_model import ISaveableModel

class Model(nn.Module):    
    def __init__(self, input_channels: int = 3):
        super().__init__()

        self._model = FastDepth(input_channels=input_channels)

    def forward(self, x):
        return self._model(x)

    def save_as(self, dir, filename):
        torch.save(self.state_dict(), join(dir, f"{filename}.pth"))

    def load_from(self, path):
        self.load_state_dict(torch.load(path))
