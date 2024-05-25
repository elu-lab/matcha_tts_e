# Github[matcha.utils.logging_utils.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/logging_utils.py

# Needs to install
# %pip install omegaconf lightning 

from typing import Any, Dict

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

# matcha_tts
from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """
    Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]
    if not trainer.logger:
        log.warning("Logger Not Found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbakcs"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
