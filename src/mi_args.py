import argparse 

def get_data_args(parser: argparse.ArgumentParser):
    # data_args: arguments related to data
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--val_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")


def get_model_args(parser: argparse.ArgumentParser):
    # model_args: arguments related to model
    parser.add_argument('--model', type=str, default='gru',
                        help='model type (default: gru)')
    parser.add_argument('--input_size', type=int, default=768, #required=True, 
                        help='size of the embeddings')
    parser.add_argument('--num_classes', type=int, default=1, #required=True,
                        help='number of classes (default: 2)')
    parser.add_argument('--num_outcomes', type=int, default=1, #required=True,
                        help='Number of outcomes (default: 1)')
    parser.add_argument('--hidden_size', type=int, default=128, #required=True,
                        help='hidden size (default: 128)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='dropout (default: 0.0)')
    parser.add_argument('--output_dropout', type=float, default=0.3,
                        help='output layer dropout (default: 0.3)')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='bidirectional (default: False)')


def get_training_args(parser: argparse.ArgumentParser):
    # training_args: arguments related to training
    parser.add_argument('--do_train', action='store_true', default=False,
                        help='do train (default: False)')
    parser.add_argument('--do_test', action='store_true', default=False,
                        help='do Test (default: False)')
    parser.add_argument('--val_folds', nargs='+', type=int, default=[],
                        help='folds to validate on (default: [])')
    parser.add_argument('--do_hparam_tune', action='store_true', default=False,
                        help='do hyperparameter tuning (default: False)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='number of trials for hyperparameter tuning (default: 100)')    
    parser.add_argument('--min_epochs', type=int, default=1, 
                        help="Minimum number of epochs for training (default: 1)")
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs to train (default: None (assumes 1000), set to -1 (inf) to train until convergence)')
    parser.add_argument("--train_batch_size", default=32, type=int, 
                        help="Batch size for training.")
    parser.add_argument("--val_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--cross_entropy_class_weight', default=None, nargs='+', type=float,
                        help='class weight for cross entropy loss (default: None)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='logging interval (default: 10)')
    parser.add_argument('--save_strategy', type=str, default='best',
                        help='model save strategy (default: best)', choices=['best', 'all'])
    parser.add_argument('--save_dir', type=str, default=None,
                        help='model save directory (default: saved_models)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--predict_last_valid_timestep', action='store_true', default=False,
                        help="predict from the last valid timestep's hidden state if true, else predict on all timestep's hidden states (default: False)")
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help="Patience for Early stopping (default: 0, ie., no early stopping)")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help="Minimum delta value for loss monitoring to consider for early stopping (default: 0.0)")
    parser.add_argument('--early_stopping_mode', type=str, default='min', choices=['min', 'max'],
                        help="Early Stopping mode (default: min)")
    #TODO: Add early stopping


def get_comet_args(parser: argparse.ArgumentParser):
    parser.add_argument('--api_key', type=str, default="~/.comet.key",
                        help='comet api key (default: ~/.comet.key)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='comet workspace (default: None)')
    parser.add_argument('--project_name', type=str, default=None,
                        help='comet project name (default: None)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='comet experiment name (default: None)')
    # parser.add_argument('--comet_save_dir', type=str, default="./lightning_logs/",
    #                     help='comet save directory (default: ./lightning_logs/)')


"""
def get_eval_args(parser: argparse.ArgumentParser):
    # eval_args: arguments related to evaluation
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='metric for evaluation (default: accuracy)')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='model load directory (default: saved_models)')
    parser.add_argument('--load_epoch', type=int, default=-1,
                        help='model load epoch (default: -1, load best model)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test (default: False)')
"""


def catch_jupyter_args(parser):
    parser.add_argument('--ip')
    parser.add_argument('--stdin')
    parser.add_argument('--control')
    parser.add_argument('--hb')
    parser.add_argument('--shell')
    parser.add_argument('--transport')
    parser.add_argument('--iopub')
    parser.add_argument('--f')
    parser.add_argument('--Session.signature_scheme')
    parser.add_argument('--Session.key')


def get_default_args(jupyter=False):
    """
        Get default arguments for Model arguments, data arguments, training arguments and comet args.
        ---
        Returns a Namespace object with the default arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Template')
    if jupyter:
        catch_jupyter_args(parser)
    get_data_args(parser)
    get_model_args(parser)
    get_training_args(parser)
    get_comet_args(parser)
    #get_eval_args(parser)
    return parser.parse_args()