import torch
import torch.nn as nn
from commons.log import log
from commons.util import exists, normpath
from torch.autograd import Variable
from torchtext import data

from dataset import create_dataset
from model import make_model, subsequent_mask

BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
UNK_WORD = '<unk>'
PAD_WORD = '<pad>'


def run(devices, dataset, model, training, **kwargs):
    CUDA_ENABLED = (devices is not None) and len(devices) > 0

    # Dataset:
    train_data, _, val_data, TGT, SRC = build_dataset(**dataset)

    # Model:
    model = make_model(len(SRC.vocab),
                       len(TGT.vocab),
                       cuda_enabled=CUDA_ENABLED,
                       **training)

    # Training:
    run_training(model=model,
                 devices=devices,
                 train_data=train_data,
                 valid_data=val_data,
                 SRC=SRC,
                 TGT=TGT,
                 cuda_enabled=CUDA_ENABLED,
                 **training)


def get_iter(dataset, batch_size, devices, train, pad_idx):
    if dataset:
        return data.Iterator(dataset,
                             batch_size=batch_size,
                             device=torch.device('cuda') if devices else None,
                             repeat=False,
                             sort_key=lambda x: (len(x.src), len(x.trg)),
                             train=train)


def run_training(model, devices, epochs, train_data, valid_data, batch_size,
                 SRC, TGT, cuda_enabled, warm_up, lr, betas, eps, log_interval,
                 **kwargs):
    pad_idx = TGT.vocab.stoi[PAD_WORD]
    criterion = LabelSmoothing(size=len(TGT.vocab),
                               padding_idx=pad_idx,
                               smoothing=0.1)
    if cuda_enabled:
        criterion.cuda()

    # Model parallelization:
    model_par = nn.DataParallel(model, device_ids=devices)

    # Optimizer:
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, warm_up,
        torch.optim.Adam(model.parameters(),
                         lr=lr,
                         betas=tuple(betas),
                         eps=eps))
    # Iterators:
    train_iter = get_iter(train_data, batch_size, devices, True, pad_idx)
    valid_iter = get_iter(valid_data, batch_size, devices, False, pad_idx)

    # Run epochs:
    for epoch in range(epochs):
        log("")
        log(("-" * 30))
        log(f"EPOCH {epoch+1}", 1)

        log("")
        log("Training...", 2)
        model_par.train()
        run_epoch(data_iter=train_iter,
                  model=model_par,
                  loss_compute=get_loss_compute(model.generator, criterion,
                                                model_opt, devices,
                                                cuda_enabled),
                  pad_idx=pad_idx,
                  log_interval=log_interval)

        if valid_iter:
            log("")
            log("Evaluating...", 2)

            # Verify loss:
            model_par.eval()
            loss = run_epoch(data_iter=valid_iter,
                             model=model_par,
                             loss_compute=get_loss_compute(
                                 model.generator, criterion, None, devices,
                                 cuda_enabled),
                             pad_idx=pad_idx,
                             log_interval=None)
            log(f"  Loss: {loss:f}", 1)

            # Verify accuracy
            accuracy = calculate_accuracy(model, valid_iter, SRC, TGT)
            log(f"  Accuracy: {accuracy:f}", 1)


def get_loss_compute(model_generator, criterion, opt, devices, cuda_enabled):
    if cuda_enabled:
        return MultiGPULossCompute(model_generator,
                                   criterion,
                                   devices=devices,
                                   opt=opt)
    else:
        return SimpleLossCompute(model_generator, criterion, opt)


def calculate_accuracy(model, test_iter, SRC, TGT):
    TO_IGNORE = [TGT.vocab.stoi[BOS_WORD], TGT.vocab.stoi[EOS_WORD]]

    def get_mask(tensor, to_ignore):
        mask = torch.ones_like(tensor).fill_(True).type_as(tensor.bool())

        for symbol in to_ignore:
            mask = mask & (tensor != symbol)
        return mask

    def matches(out, tgt):
        return (len(out) == len(tgt)) and all(out.eq(tgt))

    # Once trained we can decode the model to produce a set of translations.
    # Here we simply translate the first sentence in the validation set.
    # This dataset is pretty small so the translations with greedy search
    # are reasonably accurate.
    model.eval()

    total = 0
    correct = 0

    for i, batch in enumerate(test_iter):
        with torch.no_grad():
            src = batch.src.transpose(0, 1)
            src_mask = (src != SRC.vocab.stoi[PAD_WORD]).unsqueeze(-2)
            out = greedy_decode(model,
                                src,
                                src_mask,
                                max_len=60,
                                start_symbol=TGT.vocab.stoi[BOS_WORD],
                                end_symbol=TGT.vocab.stoi[EOS_WORD])

        tgt = batch.trg.transpose(0, 1)
        correct_items = [
            matches(o[o_mask], t[t_mask]) for (o, o_mask, t, t_mask) in zip(
                out, get_mask(out, TO_IGNORE), tgt, get_mask(tgt, TO_IGNORE))
        ]
        total += tgt.size(0)
        correct += sum(correct_items)
    accuracy = (correct / total) if (total > 0) else 0
    return accuracy


def visualize_attention(model, trans, sent):
    import matplotlib.pyplot as plt
    import seaborn

    # Attention Visualization
    # Even with a greedy decoder the translation looks pretty good. We can further
    # visualize it to see what is happening at each layer of the attention

    tgt_sent = trans.split()

    def draw(data, x, y, ax):
        seaborn.heatmap(data,
                        xticklabels=x,
                        square=True,
                        yticklabels=y,
                        vmin=0.0,
                        vmax=1.0,
                        cbar=False,
                        ax=ax)

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        log(f"Encoder Layer {layer + 1}")
        for h in range(4):
            draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
                 sent,
                 sent if h == 0 else [],
                 ax=axs[h])
        plt.show()

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        log(f"Decoder Self Layer {layer + 1}")
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[
                0, h].data[:len(tgt_sent), :len(tgt_sent)],
                 tgt_sent,
                 tgt_sent if h == 0 else [],
                 ax=axs[h])
        plt.show()
        log(f"Decoder Src Layer {layer + 1}")
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[
                0, h].data[:len(tgt_sent), :len(sent)],
                 sent,
                 tgt_sent if h == 0 else [],
                 ax=axs[h])
        plt.show()


def build_dataset(path, attributes_dir, fields, samples_min_freq,
                  max_len_sentence, train_split_ratio, vocab_min_freq,
                  **kwargs):
    """
    SRC = data.Field(sequential=True,
                     unk_token=UNK_WORD,
                     pad_token=PAD_WORD)
    TGT = data.Field(sequential=True,
                     is_target=True,
                     pad_first=True,
                     init_token=BOS_WORD,
                     eos_token=EOS_WORD,
                     unk_token=UNK_WORD,
                     pad_token=PAD_WORD)

    MAX_LEN = 100
    dataset = data.TabularDataset(
        path=f"{input_dir}\\data.json",
        format="json",
        fields={
            'frames.movement_dh_st': ('src', SRC),
            'label': ('trg', TGT)
        },
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN)
    """

    # movement_dh_st':''
    # 'movement_ndh_st':''
    # 'orientation_dh':'back'
    # 'orientation_ndh':'front'
    # 'mouth_openness':0.5540059453054104

    def compose_field(rows):
        return list(
            map(
                lambda row: "-".join([
                    f"{(row[x]['value'] if row[x] else ''):<20}"
                    for x in fields
                ]), rows))
        # return list(map(lambda row: [row[x] for x in FIELDS], rows))

    SRC = data.Field(sequential=True,
                     unk_token=UNK_WORD,
                     pad_token=PAD_WORD,
                     preprocessing=compose_field)
    TGT = data.Field(sequential=True,
                     is_target=True,
                     pad_first=True,
                     init_token=BOS_WORD,
                     eos_token=EOS_WORD,
                     unk_token=UNK_WORD,
                     pad_token=PAD_WORD)

    # Create dataset if needed:
    path = normpath(path)
    if not exists(path):
        create_dataset(attributes_dir, path, samples_min_freq)

    dataset = data.TabularDataset(
        path=path,
        format="json",
        fields={
            'phonos': ('src', SRC),
            'label': ('trg', TGT)
        },
        filter_pred=lambda x: len(vars(x)['src']) <= max_len_sentence)

    # ratios (parameter): [ train, test, val]
    # output: (train, [val,] test)
    splits = dataset.split(
        split_ratio=[train_split_ratio, 1 - train_split_ratio])

    if len(splits) == 3:
        train, val, test = splits
    else:
        train, test = splits
        val = None

    SRC.build_vocab(dataset.src, min_freq=vocab_min_freq)
    TGT.build_vocab(dataset.trg, min_freq=vocab_min_freq)

    return train, val, test, TGT, SRC


def build_static_src_vocab():
    import itertools

    x = ["", "left", "right"]
    y = ["", "up", "down"]
    z = ["", "front", "back"]

    vocab = [
        "_".join(filter(lambda v: v != "", c))
        for i, c in enumerate(itertools.product(x, y, z))
    ]
    return build_dynamic_vocab(vocab)


def build_dynamic_vocab(data):
    return set(data)


# TODO: check if can promote this to 'model' file
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5) *
                              min(step**(-0.5), step * self.warmup**(-1.5)))


# TODO: check where to move this code
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.data * norm


# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[
                Variable(o[:, i:i + chunk_size].data,
                         requires_grad=self.opt is not None)
            ] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_value = nn.parallel.gather(loss, target_device=self.devices[0])
            l_value = l_value.sum() / normalize
            total += l_value

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_value.backward()
                for j, l_value in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def run_epoch(data_iter, model, loss_compute, pad_idx, log_interval=5):
    "Standard Training and Logging Function"
    import time

    data_iter = (rebatch(pad_idx, b) for b in data_iter)
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask,
                            batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if log_interval and (i % log_interval) == 1:
            elapsed = time.time() - start
            log("  Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return (total_loss / total_tokens) if (total_tokens > 0) else 0


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, Variable(ys),
            Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        if all(next_word == end_symbol):
            break
    return ys


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
