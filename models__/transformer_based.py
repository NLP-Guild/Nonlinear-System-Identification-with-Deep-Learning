import torch
import torch.nn as nn
from models.layers import Encoder

class TransformerForRegression(nn.Module):
    def __init__(self, seq_len:int):
        self.encoder = Encoder(6,1)
        self.fc1 = nn.Linear(seq_len, seq_len*4)
        self.fc2 = nn.Linear(seq_len*4,seq_len)
        self.fc3 = nn.Linear(seq_len,2)


    def forward(self, x):
        '''

        :param x: (N,L)
        :return: (N,2)
        '''
        size = x.size()
        x = x.view((size[0],size[1],1)) # (N,L,1)
        enc_att_mask = torch.ones(size,dtype=torch.int)
        x = self.encoder(x,enc_att_mask) # (N,L,1)
        x = x.view(size) # (N,L)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



