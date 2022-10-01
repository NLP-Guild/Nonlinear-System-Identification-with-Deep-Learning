import torch
import torch.nn as nn
from models.layers import Encoder

class TransformerForRegression(nn.Module):
    def __init__(self):
        self.encoder = Encoder(6,1)