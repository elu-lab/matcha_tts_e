# Github[matcha.utils.utils.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/utils.py

# Needs to install
# !pip install wget omegaconf wandb

import os
import sys
import warnings
from importlib.util import find_spec
# What is this? Find modules
# Blog post1: https://jh-bk.tistory.com/29
# Blog post2: https://simryang.tistory.com/entry/python-%EC%84%A4%EC%B9%98%EB%90%9C-%ED%8C%A8%ED%82%A4%EC%A7%80-%ED%99%95%EC%9D%B8
# find_spec('numpy')

from pathlib import Path
from typing import Any, Callable, Dict, Tuple


import gdown
# What is this? Download files from the Google Drive with link
# Blog post: https://code-angie.tistory.com/56

import matplotlib.pyplot as plt
import numpy as np
import torch

# needs to install
import wget
from omegaconf import DictConfig

# # matcha_tts
from matcha.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


# ========================================================================================================= #

def extras(cfg: DictConfig) -> None:
    """
    Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """

    # Return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        # return # 요렇게만 가능한가?
        return 

    # Disable Python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # Pretty Print Config tree using Rich Lib
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve =True, save_to_file=True)

# ========================================================================================================= #

def task_wrapper(task_func: Callable) -> Callable:
    """
    Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """ 

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex


        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            # check if wandb is installed
            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


# ========================================================================================================= #

def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> float:
    """
    Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: The name of the metric to retrieve.
    :return: The value of the metric.
    """

    if not metric_name:
        log.info(f"Metric_name is None. Skipping Metric_value retrieval...")
        return None
    
    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
            )
        

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value. <{metric_name}={metric_value}>")

    return metric_value

# ========================================================================================================= #

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst

    # ======= example ========== #
    # lst = np.arange(1, 5)
    # item = 0
    # result = intersperse(lst, item)
    #  >> [0, 1, 0, 2, 0, 3, 0, 4, 0]

    # [item] * (len(lst) * 2 + 1)
    #  >> [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # result[1::2] = lst
    # >> [0, 1, 0, 2, 0, 3, 0, 4, 0]

    return result

# ========================================================================================================= #

def save_figure_to_numpy(fig):
    
    data = np.fromstring(fig.canvas.tostring_rgb(),
                         dtype=np.uint8, 
                         sep=""
                         ) # [3, 32, 32] -> 3*32*32
    
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # [3, 32, 32]

    return data

# ========================================================================================================= #

def plot_tensor(tensor, add_wandb_img = False):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize = (8, 3)) # (12, 3)
    im = ax.imshow(tensor, 
                   aspect = "auto", 
                   origin = "lower", 
                   interpolation = "none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()

    data = save_figure_to_numpy(fig)
    
    plt.close()

    if add_wandb_img:
        return fig
    else:
        return data


# ========================================================================================================= #

def save_plot(tensor, savepath):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize = (12, 3))
    im = ax.imshow(tensor, 
                   aspect = "auto", 
                   origin = "lower", 
                   interpolation = "none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()

# ========================================================================================================= #

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        raise TypeError(f"Unsupported type for conversion to numpy array")

# ========================================================================================================= #

def get_user_data_dir(appname = "matcha_tts"):
    """
    Args:
        appname (str): Name of Application

    Return:
        Path: path to user data directory
    """

    MATCHA_HOME = os.environ.get("MATCHA_HOME")
    if MATCHA_HOME is not None:
        ans = Path(MATCHA_HOME).expanduser().resolve(strict=False)
        # resolve(): Full Path? 
        # Blog post: https://sophiesien.tistory.com/entry/Python-Pathlib-resolve-%EC%99%80-parent
    
    # Win10, Win11 ??
    elif sys.platform == "win32":
        import winreg
        # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)

    # Apple (Mac OS): OS based on `Unix/darwin`
    elif sys.platform=="darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    
    # Linux - ubuntu ??
    else:
        ans = Path.home().joinpath(".local/share")

    final_path = ans.joinpath(appname)
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path

# ========================================================================================================= #

def assert_model_downloaded(checkpoint_path, url, use_wget=True):
    # from pathlib import Path
    if Path(checkpoint_path).exists():
        log.debug(f"[+] Model already present at {checkpoint_path}")
        print(f"[+] Model already present at {checkpoint_path}")
        return
        
    log.info(f"[-] Model Not Found at {checkpoint_path}. Will Download it")
    print(f"[-] Model Not Found at {checkpoint_path}. Will Download it")

    checkpoint_path = str(checkpoint_path)
    if not use_wget:
        gdown.download(url=url, output = checkpoint_path, quiet=False, fuzzy = True)
    else:
        wget.download(url=url, out=checkpoint_path)
        
