
from torch import Tensor
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from lightning_fabric.utilities.logger import _add_prefix


class CustomCometLogger(CometLogger):
    """
        Custom logger for Comet.ml
    """
    @rank_zero_only
    def log_metrics(self, metrics, step=None, epoch=None):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        metrics_without_epoch = metrics.copy()
        for key, val in metrics_without_epoch.items():
            if isinstance(val, Tensor):
                metrics_without_epoch[key] = val.cpu().detach()

        metrics_without_epoch = _add_prefix(metrics_without_epoch, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics_without_epoch, step=step, epoch=epoch)