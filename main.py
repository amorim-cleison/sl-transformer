from commons.log import log
from commons.util import load_args
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset


def run(args):
    # Seed:
    seed = args["seed"]
    h.setup_seed(seed)

    # Device:
    cuda = args["cuda"]
    device = h.prepare_device(cuda)

    # Dataset:
    if args["debug"]:
        args["dataset_args"]["reuse_transient"] = True
    # h.save_stats_datasets(device, args)
    dataset = AslDataset(device=device, **args).stoi()

    if args["debug"]:
        dataset = dataset.truncated(args["cv"] * 10)

    # Balance dataset:
    if should_balance_dataset(args):
        dataset = h.balance_dataset(dataset=dataset, seed=seed)
    log(f"{len(dataset)} entries of data")

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(dataset=dataset, **args)

    # Classifier:
    net_params = h.build_net_params(callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    device=device,
                                    dataset=dataset,
                                    **args)
    net = NeuralNetClassifier(**net_params)

    # Split data:
    test_size = args["test_size"]
    test_data, train_data = dataset.split(lengths=test_size,
                                          indices_only=False,
                                          seed=seed)
    log(f"> Train data: {len(train_data)} entries")
    log(f"> Test data: {len(test_data)} entries")

    # Tune hyperparams and test model:
    best_estimator = tune_hyperparams(estimator=net,
                                      callbacks_names=callbacks_names,
                                      train_data=train_data,
                                      **args)
    test_model(estimator=best_estimator, test_data=test_data, **args)


def tune_hyperparams(estimator, callbacks_names, train_data, cuda, **kwargs):
    log("\n==================== TUNING HYPERPARAMETERS ====================\n")
    phase = "grid_search"

    # Grid search:
    gs_params = h.build_grid_params(callbacks_names=callbacks_names,
                                    data=train_data,
                                    **kwargs)
    gs = GridSearchCV(estimator=estimator, **gs_params)
    log(gs_params)

    # Params grid:
    h.save_param_grid(gs.param_grid, phase=phase, **kwargs)

    # Fit:
    with parallel_backend('dask'):
        gs.fit(X=train_data.X(), y=train_data.y().to_array())

        # TODO: add something to wait dask to finish

    # Output:
    gs_output = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "best_index": int(gs.best_index_),
        "scoring": str(gs.scoring)
    }

    # Save output:
    h.save_output(gs_output, phase=phase, **kwargs)
    h.save_cv_results(gs.cv_results_, phase=phase, **kwargs)

    # Return best estimator found:
    return gs.best_estimator_


def test_model(estimator, test_data, scoring, cuda, **kwargs):
    log("\n==================== TESTING MODEL ====================\n")
    phase = "test"

    # Prepare metrics:
    if "accuracy" not in scoring:
        scoring = ["accuracy", *scoring]
    scorers = h.build_scoring(scoring=scoring, labels=test_data.labels())

    with parallel_backend('dask'):
        # Compute metrics:
        test_output = {
            f"test_{scorer.score}": scorer(estimator, test_data.X(),
                                           test_data.y().to_array())
            for scorer in scorers
        }

        # Profile model:
        with h.create_profiler(cuda) as prof:
            estimator.predict(test_data.X())

        # TODO: add something to wait dask to finish

    # Save output:
    h.save_output(test_output, phase=phase, **kwargs)
    h.save_profile(prof, phase=phase, **kwargs)


def should_balance_dataset(args):
    return ("balance_dataset" in args["dataset_args"]) and (
        args["dataset_args"]["balance_dataset"] is True)


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Dump args:
    h.dump_args(args)

    # Create distributed client:
    client = h.create_dask_client(**args)
    log(f" > Client initialized: {client}")

    # Run:
    run(args)
