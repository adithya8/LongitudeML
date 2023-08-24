import os, json
from pytorch_lightning.loggers import WandbLogger, CometLogger

def get_logger(logger_name:str, **kwargs):
    if logger_name == 'wandb':
        # Refer to https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/loggers/wandb.html#WandbLogger for detailed doc
        # name: Display name for the run.
        # save_dir/ dir: Path where data is saved. 
        # version/ id: Sets version, mainly used to resume from previous run
        # project: The name for the project.
        # prefix: A string to put at the beginning of metric keys.
        # log_model: options - latest, best, all, None, True, False
        # experiment: Wandb experiment object. Automatically set by WandbLogger.
        # offline: Run offline (data can be streamed later to wandb servers).
        # anonymous: Enables or explicitly disables anonymous logging. 
        return WandbLogger(**kwargs)
    elif logger_name == 'comet':
        # api_key: Comet API key. Get your API key from Comet.ml
        # workspace: Comet.ml workspace. If you are not using Comet.ml, leave this as None.
        # project_name: Comet.ml project name. If you are not using Comet.ml, leave this as None.
        # experiment_name: Comet.ml experiment name. If you are not using Comet.ml, leave this as None.
        # save_dir: Path where data is saved.
        if 'api_key' not in kwargs or kwargs['api_key'] is None:
            # Check if the api key file is stored in ~/.comet.key
            if os.path.exists(os.path.join(os.path.expanduser('~'), '.comet.key')):
                key_dict = json.load(open(os.path.join(os.path.expanduser('~'), '.comet.key'), 'r'))
                kwargs['api_key'] = key_dict['api_key']
            else:
                raise Warning('Comet API key not provided. Using default logger.')
        return CometLogger(**kwargs)
    else:
        raise Warning(f'Logger {logger_name} not implemented. Using default logger.')