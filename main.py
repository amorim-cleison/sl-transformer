import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset
from helper import ScoringWrapper


def run(args):
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Seed:
    h.setup_seed(**args)

    # Device:
    device = h.prepare_device(args["cuda"])

    # Dataset:
    dataset = AslDataset(device=device, batch_first=True, **args)

    if args["debug"]:
        dataset = dataset.truncated(100)

    # Balance dataset:
    dataset = h.balance_dataset(dataset)

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(**args,
                                                   **args["training_args"])

    # Cross-validator:
    cross_validator = h.get_cross_validator(**args)

    # Classifier:
    net_params = h.build_net_params(callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    device=device,
                                    dataset=dataset,
                                    cross_validator=cross_validator,
                                    **args)
    net = NeuralNetClassifier(**net_params)

    # Train:
    if args["mode"] == "train":
        run_training(net=net,
                     dataset=dataset,
                     cross_validator=cross_validator,
                     **args,
                     **args["training_args"])

    # Grid search:
    elif args["mode"] == "grid":
        run_grid_search(net=net,
                        callbacks_names=callbacks_names,
                        dataset=dataset,
                        cross_validator=cross_validator,
                        **args)


def run_training(net, dataset, cross_validator, scoring, **kwargs):
    if cross_validator:
        run_training_cv(net=net,
                        dataset=dataset,
                        cross_validator=cross_validator,
                        scoring=scoring)
    else:
        run_training_no_cv(net, dataset, scoring)


def run_training_no_cv(net, dataset, scoring):
    _scorer = get_scorer(scoring)

    # Data split:
    test_ds, train_ds = dataset.split(0.2)
    y_test = test_ds.y(collated=True)

    # Fit:
    net.fit(train_ds, None)

    # Scoring:
    score = _scorer(net, test_ds, y_test.cpu())
    print(f"Test accuracy: {score:.3f}")


def run_training_cv(net, dataset, cross_validator, scoring):
    # Cross-validation:
    scores = cross_val_score(net,
                             dataset.X(),
                             dataset.y(),
                             cv=cross_validator,
                             scoring=ScoringWrapper(scoring),
                             error_score='raise')

    print(
        f"'{scoring.capitalize()}' per fold: {[f'{x:.3f}' for x in scores]}]")
    print(f"Avg validation '{scoring}': {scores.mean():.3f}")


def run_grid_search(net, callbacks_names, dataset, cross_validator, **kwargs):
    # Grid search:
    grid_params = h.build_grid_params(callbacks_names=callbacks_names,
                                      cross_validator=cross_validator,
                                      **kwargs)
    gs = GridSearchCV(net, **grid_params)
    gs.fit(dataset.X(), dataset.y())

    # Output:
    gs_output = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "best_index": int(gs.best_index_),
        "scoring": str(gs.scoring)
    }
    print(gs_output)

    # Save output:
    log("Saving grid search output...")
    save_json(data=gs_output, path=f"{args['workdir']}/grid_search.json")
    pd.DataFrame(
        gs.cv_results_).to_csv(f"{args['workdir']}/grid_search_results.csv")


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    run(args)
