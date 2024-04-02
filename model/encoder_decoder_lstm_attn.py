from model.base import EncoderDecoderAttnBase


class EncoderDecoderLSTMAttn(EncoderDecoderAttnBase):
    def __init__(self, **kwargs):
        super(EncoderDecoderLSTMAttn, self).__init__(rnn_type="lstm", **kwargs)
