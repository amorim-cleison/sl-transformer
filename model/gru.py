import torch.nn as nn

from .base import RNNModel


class GRU(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        super(GRU, self).__init__(rnn=rnn, model_type="GRU", **kwargs)

    def build(self, input_size, hidden_size, num_layers, dropout, **kwargs):
        return nn.GRU(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout)
