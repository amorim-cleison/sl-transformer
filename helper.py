import random
from pydoc import locate

import numpy as np
import pandas as pd
import torch
from commons.log import log
from commons.util import (create_if_missing, normpath, save_args, save_items,
                          save_json)
from sklearn.model_selection import *
from skorch.callbacks import (Checkpoint, EarlyStopping, EpochScoring,
                              GradientNormClipping, LRScheduler)
from skorch.dataset import ValidSplit
from torch.profiler import profile

import model.util as util
from dataset import AslDataset


def setup_seed(seed, **kwargs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            log("WARNING: CUDA device(s) available, so you should "
                "probably run with --cuda")

    return torch.device("cuda" if cuda else "cpu")


def dump_args(args):
    _dir = args['workdir']
    create_if_missing(_dir)
    save_args(args, normpath(f"{_dir}/config.yaml"))


def build_net_params(model_args, model, optimizer, criterion, callbacks,
                     callbacks_names, device, dataset, optimizer_args,
                     criterion_args, cv, **kwargs):
    src_vocab = dataset.vocab_X
    tgt_vocab = dataset.vocab_y

    # Callbacks args:
    _callbacks_args = build_callbacks_args(model=model,
                                           callbacks_names=callbacks_names,
                                           **kwargs)

    # Module args:
    _module_args = prefix_args("module",
                               ensure_list=False,
                               batch_first=dataset.batch_first,
                               src_vocab=src_vocab,
                               tgt_vocab=tgt_vocab,
                               device=device,
                               **model_args)

    # Module args:
    _optimizer_args = prefix_args("optimizer",
                                  ensure_list=False,
                                  **optimizer_args)

    # Criterion args:
    criterion_args["ignore_index"] = util.get_pad_idx(tgt_vocab)
    _criterion_args = prefix_args("criterion",
                                  ensure_list=False,
                                  **criterion_args)

    # Iterators args:
    iterators_args = {
        "collate_fn": collate_data,
        # "num_workers": 4,
        # "shuffle": True
    }
    _iterator_train_args = prefix_args("iterator_train",
                                       ensure_list=False,
                                       **iterators_args)
    _iterator_valid_args = prefix_args("iterator_valid",
                                       ensure_list=False,
                                       **iterators_args)
    # Other args:
    KEYS_FOR_NET = [
        "lr", "max_epochs", "batch_size", "predict_nonlinearity", "warm_start",
        "verbose"
    ]
    _net_args = filter_by_keys(kwargs, keys_to_filter=KEYS_FOR_NET)

    return {
        "device": device,
        "module": locate(model),
        "optimizer": locate(optimizer),
        "criterion": locate(criterion),
        "callbacks": callbacks,
        "dataset": AslDataset,
        "train_split": ValidSplit(cv),
        **_net_args,
        **_module_args,
        **_optimizer_args,
        **_criterion_args,
        **_callbacks_args,
        **_iterator_train_args,
        **_iterator_valid_args
    }


def build_grid_params(grid_args, callbacks_names, model, workdir, scoring,
                      verbose, n_jobs, cv, data, **kwargs):
    def __unpack_args(callbacks_names,
                      model,
                      workdir,
                      scoring,
                      verbose,
                      n_jobs,
                      data,
                      cv,
                      training_args={},
                      model_args={},
                      optimizer_args={},
                      criterion_args={},
                      **kwargs):
        # Callbacks args:
        _callbacks_args = build_callbacks_args(model=model,
                                               workdir=workdir,
                                               callbacks_names=callbacks_names,
                                               ensure_list=True,
                                               **kwargs)

        # Module args:
        _module_args = prefix_args("module", ensure_list=True, **model_args)

        # Optimizer args:
        _optimizer_args = prefix_args("optimizer",
                                      ensure_list=True,
                                      **optimizer_args)

        # Criterion args:
        _criterion_args = prefix_args("criterion",
                                      ensure_list=True,
                                      **criterion_args)

        # General args:
        _kwargs = prefix_args(None, ensure_list=True, **kwargs)

        # Other args:
        KEYS_FOR_GRID = [
            "n_jobs", "refit", "verbose", "pre_dispatch", "return_train_score"
        ]
        _grid_args = filter_by_keys(kwargs, keys_to_filter=KEYS_FOR_GRID)

        # Scoring:
        labels = data.labels()
        _scoring_wrapper = build_scoring(scoring, labels, allow_multiple=False)

        return {
            "refit": True,
            "cv": cv,
            "verbose": verbose,
            "scoring": _scoring_wrapper,
            "n_jobs": n_jobs,
            "error_score": "raise",
            **_grid_args, "param_grid": {
                **_module_args,
                **_optimizer_args,
                **_criterion_args,
                **_callbacks_args,
                **_kwargs
            }
        }

    return __unpack_args(callbacks_names=callbacks_names,
                         model=model,
                         workdir=workdir,
                         scoring=scoring,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         data=data,
                         cv=cv,
                         **grid_args)


def build_test_params(scoring, verbose, n_jobs, cv, data, **kwargs):
    # Scoring:
    labels = data.labels()
    _scoring_wrapper = build_scoring(scoring, labels, allow_multiple=False)

    return {
        "cv": cv,
        "n_jobs": n_jobs,
        "verbose": verbose,
        "scoring": _scoring_wrapper,
        "error_score": "raise",
    }


def build_callbacks(mode,
                    workdir,
                    scoring,
                    dataset,
                    early_stopping=None,
                    gradient_clipping=None,
                    lr_scheduler=None,
                    **kwargs):
    monitor = "valid"

    # Callbacks:
    callbacks = []

    # Checkpoint saving:
    checkpoint = Checkpoint(monitor=f"{monitor}_loss_best", dirname=workdir)
    callbacks.append(("checkpoint", checkpoint))

    # # Init state (resume):
    # if resumable:
    #     callbacks.append(("resume_state", LoadInitState(checkpoint)))

    # Early stopping:
    if early_stopping:
        callbacks.append(("early_stopping",
                          EarlyStopping(**early_stopping,
                                        monitor=f"{monitor}_loss",
                                        lower_is_better=True,
                                        sink=log)))

    # Gradient clip:
    if gradient_clipping:
        callbacks.append(
            ("gradient_clipping", GradientNormClipping(**gradient_clipping)))

    # LR Scoring:
    def lr_score(net, X, y=None):
        return net.optimizer_.param_groups[0]["lr"]

    callbacks.append(
        ("lr_scoring", EpochScoring(scoring=lr_score,
                                    name='lr',
                                    on_train=False)))

    # LR Scheduler:
    if lr_scheduler:
        callbacks.append(("lr_scheduler",
                          LRScheduler(monitor=f"{monitor}_loss",
                                      step_every="epoch",
                                      **lr_scheduler)))

    # Scoring metric (dynamic):
    if not isinstance(scoring, list):
        scoring = [scoring]

    labels = dataset.labels()
    scoring_wrappers = build_scoring(scoring, labels, allow_multiple=True)

    for wrapper in scoring_wrappers:
        # Valid:
        callbacks.append(
            (f"score_valid_{wrapper.score}",
             EpochScoring(scoring=wrapper,
                          name=f"valid_{wrapper.score}",
                          on_train=False,
                          lower_is_better=not wrapper.greater_is_better)))
        # Train:
        callbacks.append(
            (f"score_train_{wrapper.score}",
             EpochScoring(scoring=wrapper,
                          name=f"train_{wrapper.score}",
                          on_train=True,
                          lower_is_better=not wrapper.greater_is_better)))

    # Callbacks names:
    callbacks_names = [c[0] for c in callbacks]

    return callbacks, callbacks_names


def build_scoring(scoring, labels=None, allow_multiple=True):
    if not isinstance(scoring, list):
        scoring = [scoring]
    wrappers = [ScoringWrapper(score, labels) for score in scoring]

    if not allow_multiple:
        wrappers = wrappers[0]
    return wrappers


def build_callbacks_args(callbacks_names, ensure_list=False, **kwargs):
    __standard_callbacks = ["print_log"]
    callbacks_args = filter_by_keys(kwargs,
                                    callbacks_names + __standard_callbacks)
    return prefix_args("callbacks", ensure_list=ensure_list, **callbacks_args)


def collate_data(data):
    X, y = zip(*data)

    if len(X[0]) == 3:
        X, X_lengths, _ = zip(*X)
    elif len(X[0]) == 2:
        X, X_lengths = zip(*X)

    X = torch.tensor(X, dtype=torch.long)
    X_lengths = torch.tensor(X_lengths, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return {"X": X, "lengths": X_lengths, "y": y}, y


def format_dir(dir, **kwargs):
    from datetime import datetime
    params = {
        "datetime": datetime.now(),
        **kwargs,
    }
    return normpath(dir.format(**params)) if dir is not None else ''


def filter_by_keys(map, keys_to_filter, not_in=False):
    return dict(
        filter(lambda o: not (o[0] in keys_to_filter) == not_in, map.items()))


def is_cv_for_net(cross_validator):
    return isinstance(cross_validator, (int, float, ValidSplit))


def prefix_args(prefix, ensure_list=False, output=None, **kwargs):
    if output is None:
        output = {}

    for k, v in kwargs.items():
        name = k if prefix is None else f"{prefix}__{k}"

        if isinstance(v, dict):
            prefix_args(prefix=name,
                        output=output,
                        ensure_list=ensure_list,
                        **v)
        else:
            if ensure_list and not isinstance(v, list):
                v = [v]
            output[name] = v
    return output


def balance_dataset(dataset, seed):
    import math
    from collections import Counter
    from statistics import mean

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler

    from dataset import AslDataset

    def compute_sampling(data, u, mode="under"):
        def smooth_v(v, u, sign):
            tmp = round(u + math.log(v))
            return v if (v * sign) > (tmp * sign) else tmp

        _signs = {"under": -1, "over": +1}
        assert (mode in _signs), "Invalid mode"

        sign = _signs[mode]
        return {k: smooth_v(v, u, sign) for (k, v) in data.items()}

    # Original data:
    X, y = dataset.X().to_array(), dataset.y().to_array()
    original = Counter(y)
    u = mean(original.values())

    # Compute samplings:
    under_sampling = compute_sampling(original, u, "under")
    over_sampling = compute_sampling(under_sampling, u, "over")

    # Prepare pipeline:
    rus = RandomUnderSampler(sampling_strategy=under_sampling,
                             random_state=seed,
                             replacement=False)
    ros = RandomOverSampler(sampling_strategy=over_sampling, random_state=seed)
    pipeline = Pipeline(steps=[(type(rus).__name__, rus),
                               (type(ros).__name__, ros)],
                        verbose=True)

    # Resample data:
    X_res, y_res = pipeline.fit_resample(X, y)
    dataset_res = AslDataset(dataset=dataset, X=X_res, y=y_res)

    return dataset_res


def create_profiler(cuda):
    return profile(record_shapes=False,
                   profile_memory=True,
                   with_flops=True,
                   with_stack=False,
                   with_modules=False)


def save_stats_datasets(device, args):
    from collections import Counter

    from commons.util import save_json

    ds = AslDataset(device=device, batch_first=True, **args)
    labels = ds._AslDataset__data[:][1]
    cnt = Counter(labels)
    save_json(dict(cnt), "./tmp.json")

    ds_bal = balance_dataset(dataset=ds, seed=args["seed"])
    labels_bal = ds_bal._AslDataset__data[:][1]
    cnt_bal = Counter(labels_bal)
    save_json(dict(cnt_bal), "./tmp_bal.json")


def save_param_grid(grid_params, phase, workdir, **kwargs):
    import itertools

    log("Saving grid params...")

    vals = [x for x in grid_params.values()]
    cols = [x for x in grid_params.keys()]
    cross_product = list(itertools.product(*vals))

    df_param_grid = pd.DataFrame(cross_product, columns=cols)
    log(df_param_grid)
    df_param_grid.to_csv(f"{workdir}/{phase}_grid_params.csv")


def save_cv_results(cv_results, phase, workdir, **kwargs):
    log("Saving CV results...")
    df_cv_results = pd.DataFrame(cv_results)
    log(df_cv_results)
    df_cv_results.to_csv(f"{workdir}/{phase}_results.csv")


def save_output(output, phase, workdir, **kwargs):
    log("Saving output...")
    log(output)
    save_json(output, f"{workdir}/{phase}_output.json")


def save_profile(profiler, phase, workdir, **kwargs):
    log("Saving profile...")
    key_averages = profiler.key_averages()

    # Table:
    table = key_averages.table(sort_by="self_cpu_time_total",
                               top_level_events_only=True)
    save_items([table], f"{workdir}/{phase}_profile_table.txt")

    # Details:
    total_average = key_averages.total_average()
    details = {
        # CPU:
        "cpu_memory_usage": total_average.cpu_memory_usage,
        "cpu_time": total_average.cpu_time,
        "cpu_time_str": total_average.cpu_time_str,
        "cpu_time_total": total_average.cpu_time_total,
        "cpu_time_total_str": total_average.cpu_time_total_str,
        "self_cpu_memory_usage": total_average.self_cpu_memory_usage,
        "self_cpu_time_total": total_average.self_cpu_time_total,
        "self_cpu_time_total_str": total_average.self_cpu_time_total_str,

        # CUDA:
        "cuda_memory_usage": total_average.cuda_memory_usage,
        "cuda_time": total_average.cuda_time,
        "cuda_time_str": total_average.cuda_time_str,
        "cuda_time_total": total_average.cuda_time_total,
        "cuda_time_total_str": total_average.cuda_time_total_str,
        "self_cuda_memory_usage": total_average.self_cuda_memory_usage,
        "self_cuda_time_total": total_average.self_cuda_time_total,
        "self_cuda_time_total_str": total_average.self_cuda_time_total_str,

        # FLOPS:
        "flops": total_average.flops,

        # Others:
        "device_type": total_average.device_type.name,
        "count": total_average.count,
        "input_shapes": str(total_average.input_shapes),
        "scope": str(total_average.scope),
        "is_legacy": str(total_average.is_legacy),
        "is_remote": str(total_average.is_remote),
        "is_async": str(total_average.is_async),
    }
    log(details)
    save_json(details, f"{workdir}/{phase}_profile.json")


def create_dask_client(dask_args, **kwargs):
    import os

    from dask.distributed import Client

    log("Initializing Dask client...")

    # Parameters:
    scheduler = str(dask_args.get("scheduler", ""))
    source = str(dask_args.get("source", ""))
    node = str(dask_args.get("node", "localhost"))
    cpus_per_task = int(dask_args.get("cpus_per_task", os.cpu_count()))

    if scheduler:
        client = Client(address=scheduler)
    else:
        # FIXME: and linux
        if (torch.cuda.is_available()):
            from dask_cuda import LocalCUDACluster
            gpus = os.getenv("CUDA_VISIBLE_DEVICES")
            cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=gpus,
                                       name=f"cluster-{node}-gpu{gpus}",
                                       threads_per_worker=cpus_per_task)
        else:
            from dask.distributed import LocalCluster
            cluster = LocalCluster(name=f"cluster-{node}-cpu",
                                   threads_per_worker=cpus_per_task,
                                   processes=False)

        client = Client(cluster)

    # Upload source code:
    if source:
        log(f" > Uploading '{source}' to client...")
        client.upload_file(normpath(source))

    return client


class ScoringWrapper:
    def __init__(self, score_func, labels=None):
        from sklearn.metrics import get_scorer
        self._score_func = score_func
        self.scorer = get_scorer(score_func)
        # FIXME: add support to externalize scoring kwargs/options:
        if (score_func == 'neg_log_loss'):
            self.scorer._kwargs["labels"] = labels
        elif (score_func == 'accuracy'):
            pass
        else:
            self.scorer._kwargs["zero_division"] = 0

    def __call__(self, estimator, X, y_true, sample_weight=None):
        return self.scorer(estimator, X, y_true, sample_weight)

    def __repr__(self):
        return f"{type(self).__name__}('{self._score_func}')"

    @property
    def greater_is_better(self):
        return (self.scorer._sign == 1)

    @property
    def score(self):
        return self._score_func
