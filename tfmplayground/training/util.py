from functools import wraps
import logging
from typing import Tuple
import os

import torch
from torch.distributed import destroy_process_group
from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_ddp_availability() -> bool:
    """
    Check whether DDP is available.
    """

    ddp_available = torch.distributed.is_available() and (torch.cuda.device_count() > 1)
    if not ddp_available:
        return ddp_available

    assert int(os.environ.get("WORLD_SIZE", 0)) <= torch.cuda.device_count(), (
        f"Number of GPUs ({torch.cuda.device_count()}) is less than the number of processes \
({os.environ.get('WORLD_SIZE', 0)})"
    )

    if int(os.environ.get("WORLD_SIZE", 0)) != torch.cuda.device_count():
        raise RuntimeError(
            f"Number of GPUs available ({torch.cuda.device_count()}) is not equal to the number of processes \
({os.environ.get('WORLD_SIZE', 0)}). Please specify torchrun --nproc-per-node=NUM_GPUS. If you have more GPUs \
on your machine, then you want to use, set CUDA_VISIBLE_DEVICES."
        )

    return ddp_available

def infer_device(use_cpu: bool) -> Tuple[str, bool]:
    """
    Automatically infer the device to use for training. If DDP is availeble,
    this method will automatically setup DDP for training.

    Parameters
    ----------
    use_cpu: bool
        Force the use of CPU. If no CUDA is available CPU will automatically
        be used.

    Returns
    -------
    Tuple[str, bool]
        - The device to use for training. Either "cuda" or "cpu".
        - Whether DDP is available and used.
    """

    device = "cuda"
    ddp = False
    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
        logger.info("Using CPU for training.")
        return device, ddp

    ddp = check_ddp_availability()

    return device, ddp

def ddp_teardown(func):
    """
    Decorator to ensure ddp process group is cleaned up even if training fails.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                print("Cleaning up ddp process group...")
                destroy_process_group()

    return wrapper

def log_on_main(logger: logging.Logger, message: str, level: int) -> None:
    """
    Simple function to log only on main process in ddp setting.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.log(level, message)

def tqdm_on_main(message: str) -> None:
    """
    tqdm write only on main process in ddp setting.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tqdm.write(message)