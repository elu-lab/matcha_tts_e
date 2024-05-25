# Github[matcha.utils.instantiators.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/instantiators.py

# Needs to install
# pip install lightning omegaconf
# pip install hydra-core --upgrade
import time
from typing import List

# hydra (Meta)
import hydra

# lightning
from lightning import Callback
from lightning.pytorch.loggers import Logger

# omegaconf
from omegaconf import DictConfig

# matcha_tts
from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:

    """
    Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No Callback configs Found! Skipping...")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating Callback <{cb_conf._target_}>")
             # pylint: disable=protected-access

            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

# ============================================================================================= #

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:

    """
    Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """

    logger: List[Logger] = []
    time.sleep(.1)

    if not logger_cfg:
        log.warning("No logger configs Found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
             # pylint: disable=protected-access

            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

