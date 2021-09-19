import torch
from dataset.constant import PAD_WORD


def get_pad_idx(vocab):
    return vocab.stoi[PAD_WORD]


def generate_mask(data):
    """
    Mask ensures that position i is allowed to attend the unmasked
    positions. If a ByteTensor is provided, the non-zero positions are
    not allowed to attend while the zero positions will be unchanged.
    If a BoolTensor is provided, positions with ``True`` are not
    allowed to attend while ``False`` values will be unchanged.
    If a FloatTensor is provided, it will be added to the attention
    weight.
    """
    def generate_square_subsequent_mask(sz: int):
        r"""
        Generate a square mask for the sequence. The masked positions are
        filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Extracted from `torch.nn.Transformer.generate_square_subsequent_mask`.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    mask = generate_square_subsequent_mask(data.size(0))
    mask = (mask != float(0.0)).bool()
    return mask


def generate_padding_mask(data, vocab):
    """
    Padding mask provides specified elements in the key to be ignored
    by the attention. If a ByteTensor is provided, the non-zero
    positions will be ignored while the zero positions will be
    unchanged. If a BoolTensor is provided, the positions with the
    value of ``True`` will be ignored while the position with the
    value of ``False`` will be unchanged.
    """
    pad_idx = get_pad_idx(vocab)
    mask = (data == pad_idx).transpose(0, 1).bool()
    return mask


def generate_seq_lengths(data, vocab):
    pad_idx = get_pad_idx(vocab)
    return (data != pad_idx).transpose(0, 1).sum(dim=-1)


def pad_to_shape(data, shape, vocab, pad_first=False):
    import torch.nn.functional as F
    pad_len = shape[0] - data.shape[0]

    if pad_len > 0:
        pad_idx = get_pad_idx(vocab)
        pad_first_len = pad_len if pad_first else 0
        pad_last_len = 0 if pad_first else pad_len
        pad = (0, 0, pad_first_len, pad_last_len)
        return F.pad(input=data, pad=pad, mode="constant", value=pad_idx)
    else:
        return data