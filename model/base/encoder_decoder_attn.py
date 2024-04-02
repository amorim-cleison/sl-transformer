# -----------------------------------------------------------
# This implementation was taken from (with some minor adjustments):
#
# J Bastings. 2018. The Annotated Encoder-Decoder with Attention.
# https://bastings.github.io/annotated_encoder_decoder/
# -----------------------------------------------------------
import model.util as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self,
                 encoder,
                 decoder,
                 src_embed,
                 trg_embed,
                 generator,
                 max_output_len=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        self.max_output_len = max_output_len

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths,
                                                    self.src_embed.padding_idx)
        # return self.decode(encoder_hidden, encoder_final, src_mask, trg,
        #                    trg_mask)
        out, _, _ = self.decode(encoder_hidden,
                                encoder_final,
                                src_mask,
                                trg,
                                trg_mask,
                                max_len=self.max_output_len)
        return self.generator(out)

    def encode(self, src, src_mask, src_lengths, src_padding_idx):
        return self.encoder(self.src_embed(src), src_mask, src_lengths,
                            src_padding_idx)

    def decode(self,
               encoder_hidden,
               encoder_final,
               src_mask,
               trg,
               trg_mask,
               decoder_hidden=None,
               max_len=None):
        return self.decoder(self.trg_embed(trg),
                            encoder_hidden,
                            encoder_final,
                            src_mask,
                            trg_mask,
                            hidden=decoder_hidden,
                            max_len=max_len)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_class,
                 num_layers=1,
                 dropout=0.,
                 batch_first=True):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn = rnn_class(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.)

    def forward(self, X, mask, lengths, padding_idx):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        total_length = X.size(1)

        packed = pack_padded_sequence(X,
                                      lengths.cpu(),
                                      batch_first=self.batch_first,
                                      enforce_sorted=False)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output,
                                        batch_first=self.batch_first,
                                        total_length=total_length,
                                        padding_value=padding_idx)

        # we need to manually concatenate the final states for both directions
        hidden_concat = self.concatenate_directions(hidden)

        return output, hidden_concat

    def concatenate_directions(self, hidden):
        fwd_hidden = hidden[0:hidden.size(0):2]
        bwd_hidden = hidden[1:hidden.size(0):2]
        hidden_concat = torch.cat([fwd_hidden, bwd_hidden],
                                  dim=2)  # [num_layers, batch, 2*dim]
        return hidden_concat


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    def __init__(self,
                 emb_size,
                 hidden_size,
                 attention,
                 rnn_class,
                 num_layers=1,
                 dropout=0.5,
                 bridge=True,
                 batch_first=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.batch_first = batch_first

        self.rnn = rnn_class(input_size=(emb_size + 2 * hidden_size),
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             dropout=dropout if num_layers > 1 else 0.)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size,
                                bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(
            (hidden_size + 2 * hidden_size + emb_size),
            hidden_size,
            bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key,
                     hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        # query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        query = self.get_query(hidden)
        context, attn_probs = self.attention(query=query,
                                             proj_key=proj_key,
                                             value=encoder_hidden,
                                             mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(
            self,
            trg_embed,
            encoder_hidden,  # encoder_output - hidden states from the encoder, shape (batch_size, src_length, encoder.output_size)
            encoder_final,  # encoder_hidden - last state from the encoder, shape (batch_size, encoder.output_size)
            src_mask,
            trg_mask,
            hidden=None,
            max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            output, hidden, pre_output = self.forward_step(
                prev_embed=prev_embed,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                proj_key=proj_key,
                hidden=hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        # return torch.tanh(self.bridge(encoder_final))
        hidden = torch.tanh(self.bridge(encoder_final))

        if isinstance(self.rnn, nn.LSTM):
            hidden = (hidden, hidden)
        return hidden

    def get_query(self, hidden):
        if isinstance(hidden, tuple):
            hidden, _ = hidden
        return hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


class EncoderDecoderAttnBase(nn.Module):

    MAX_OUTPUT_LEN = 1

    RNN_CLASSES = {"gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 batch_first,
                 rnn_type,
                 embedding_size=256,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.1,
                 **kwargs):
        super(EncoderDecoderAttnBase, self).__init__()
        assert (rnn_type in self.RNN_CLASSES), "Invalid `rnn_type`."
        rnn_class = self.RNN_CLASSES[rnn_type]

        self.batch_first = batch_first
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Pad index:
        src_pad_idx = util.get_pad_idx(src_vocab)
        tgt_pad_idx = util.get_pad_idx(tgt_vocab)

        # Vocab size:
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)

        self.model = EncoderDecoder(
            Encoder(input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    rnn_class=rnn_class,
                    batch_first=batch_first),
            Decoder(emb_size=embedding_size,
                    hidden_size=hidden_size,
                    attention=BahdanauAttention(hidden_size),
                    num_layers=num_layers,
                    dropout=dropout,
                    rnn_class=rnn_class,
                    batch_first=batch_first),
            nn.Embedding(num_embeddings=src_vocab_size,
                         embedding_dim=embedding_size,
                         padding_idx=src_pad_idx),
            nn.Embedding(num_embeddings=tgt_vocab_size,
                         embedding_dim=embedding_size,
                         padding_idx=tgt_pad_idx),
            Generator(hidden_size=hidden_size, vocab_size=tgt_vocab_size),
            max_output_len=self.MAX_OUTPUT_LEN)

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return super().to(device)

    def forward(self, X, y, lengths, **kwargs):
        src = X
        tgt = self.prepend_bos(y, self.tgt_vocab)

        # Masks:
        src_mask = self.generate_mask(src, self.src_vocab)
        tgt_mask = self.generate_mask(tgt, self.tgt_vocab)

        # Lengths:
        src_lengths = lengths
        tgt_lengths = util.resolve_lengths(tgt, self.tgt_vocab)

        output = self.model(src, tgt, src_mask, tgt_mask, src_lengths,
                            tgt_lengths)
        return output[:, -1]

    def generate_mask(self, data, vocab):
        pad_idx = util.get_pad_idx(vocab)
        return (data != pad_idx).unsqueeze(1)

    def prepend_bos(self, data, vocab):
        batch_size = data.size(0)
        data = data.unsqueeze(1)
        bos_idx = util.get_bos_idx(vocab)
        bos_data = torch.full((batch_size, 1), bos_idx).to(self.device)
        return torch.cat([bos_data, data], dim=1)
