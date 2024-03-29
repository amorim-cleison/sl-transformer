import torch
from dataset.constant import PAD_WORD, BOS_WORD


def get_pad_idx(vocab):
    return vocab.stoi[PAD_WORD]

def get_bos_idx(vocab):
    return vocab.stoi[BOS_WORD]

def generate_mask(data, batch_first=False):
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

    features_dim = 1 if batch_first else 0

    if data.ndim == 1:
        data = data.unsqueeze(features_dim)

    size = data.size(features_dim)
    mask = generate_square_subsequent_mask(size)
    mask = (mask != float(0.0)).bool()
    return mask


def generate_padding_mask(data, vocab, batch_first=False):
    """
    Padding mask provides specified elements in the key to be ignored
    by the attention. If a ByteTensor is provided, the non-zero
    positions will be ignored while the zero positions will be
    unchanged. If a BoolTensor is provided, the positions with the
    value of ``True`` will be ignored while the position with the
    value of ``False`` will be unchanged.
    """
    pad_idx = get_pad_idx(vocab)
    mask = (data == pad_idx).bool()

    if mask.ndim < 2:
        mask = mask.unsqueeze(-1)
    return mask


def resolve_lengths(data, vocab, dim=-1):
    pad_idx = get_pad_idx(vocab)

    if (data.ndim < 2):
        data = data.unsqueeze(dim)
    return data.size(dim) - data.eq(pad_idx).sum(dim=dim)
