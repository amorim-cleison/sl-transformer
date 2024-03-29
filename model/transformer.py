import math

import torch.nn as nn

from .component import PositionalEncoding
from .util import generate_mask, generate_padding_mask


class Transformer(nn.Module):
    def __init__(self,
                 embedding_size,
                 num_heads,
                 num_layers,
                 hidden_size,
                 dropout,
                 src_vocab,
                 tgt_vocab,
                 device=None,
                 batch_first=False,
                 **kwargs):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.embedding_size = embedding_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.batch_first = batch_first

        input_size = len(src_vocab)
        output_size = len(tgt_vocab)

        self.src_embedding = nn.Embedding(num_embeddings=input_size,
                                          embedding_dim=embedding_size)
        self.src_pos_encoding = PositionalEncoding(d_model=embedding_size,
                                                   dropout=dropout)
        self.tgt_embedding = nn.Embedding(num_embeddings=output_size,
                                          embedding_dim=embedding_size)
        self.tgt_pos_encoding = PositionalEncoding(d_model=embedding_size,
                                                   dropout=dropout)
        self.transformer = nn.Transformer(d_model=embedding_size,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=hidden_size,
                                          dropout=dropout,
                                          batch_first=batch_first)
        self.linear = nn.Linear(in_features=embedding_size,
                                out_features=output_size)
        self.softmax = nn.functional.log_softmax

    def to(self, device):
        self.src_embedding = self.src_embedding.to(device)
        self.src_pos_encoding = self.src_pos_encoding.to(device)
        self.tgt_embedding = self.tgt_embedding.to(device)
        self.tgt_pos_encoding = self.tgt_pos_encoding.to(device)
        self.transformer = self.transformer.to(device)
        self.linear = self.linear.to(device)
        self.device = device
        return super().to(device)

    def forward(self, X, y, lengths=None, **kwargs):
        assert (X is not None), "`X` is a required paramenter"
        assert (y is not None), "`y` is a required paramenter"

        src = self.adjust_batch_in(X)
        tgt = self.adjust_batch_in(y)

        # Masks:
        src_mask = generate_mask(src, self.batch_first).to(self.device)
        tgt_mask = generate_mask(tgt, self.batch_first).to(self.device)
        src_padding_mask = \
            generate_padding_mask(src, self.src_vocab, self.batch_first).to(self.device)
        tgt_padding_mask = \
            generate_padding_mask(tgt, self.tgt_vocab, self.batch_first).to(self.device)

        # Embeddings:
        src_embed = self.forward_embedding(src, self.src_embedding,
                                           self.src_pos_encoding)
        tgt_embed = self.forward_embedding(tgt, self.tgt_embedding,
                                           self.tgt_pos_encoding)

        # Forward:
        output = self.transformer(src=src_embed,
                                  tgt=tgt_embed,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        output = self.linear(output)
        output = self.softmax(output, dim=-1)
        return self.adjust_batch_out(output)

    def adjust_batch_in(self, data):
        if data.ndim < 2:
            data = data.unsqueeze(-1)
        return data

    def adjust_batch_out(self, data):
        if data.ndim > 2:
            features_dim = 1 if self.batch_first else 0
            data = data.squeeze(dim=features_dim)
        return data

    def forward_embedding(self, x, embedding, pos_encoding):
        x = embedding(x) * math.sqrt(self.embedding_size)
        x = pos_encoding(x)
        return x
