# from sl_transformer import run
from sl_transformer_torch import run
from commons.util import Argument, load_args
from commons.log import init_logger

ARGUMENTS = [
    Argument('-d', '--debug', type=bool, help='Debug flag'),
    Argument('-s',
             '--seed',
             type=int,
             default=1,
             help='Random seed for reproducibility'),
    Argument('-nv', '--cuda', type=bool, default=False, help='Enable cuda'),
    # Argument('-de', '--devices', type=int, help='# of available GPUs to use'),
    Argument('-ds', '--dataset', type=dict, help='Options for the dataset'),
    Argument('-md', '--model', type=dict, help='Options for the model'),
    Argument('-tr', '--training', type=dict, help='Options for the training'),
    Argument('-tl',
             '--transfer_learning',
             type=dict,
             help='Options for transfering learning'),
]

if __name__ == "__main__":
    args = load_args('SL Transformer', ARGUMENTS)
    init_logger(args)

    args = {
        f"{k}_args" if isinstance(v, dict) else k: v
        for (k, v) in vars(args).items()
    }
    run(**args)
