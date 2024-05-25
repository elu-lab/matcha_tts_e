"""
Github[bastligthingmodule.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/baselightningmodule.py

This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC 
from typing import Any, Dict

import wandb

import torch

#lightning
import lightning
# import lightning as L

from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics= {"mel_mean": 0.0, "mel_std": 1.0}

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params = self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams,scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch

            return {"optimizer": optimizer, 
                    "lr_scheduler": { "scheduler": scheduler, 
                                     "interval": self.hparams.scheduler.lightning_args.interval,
                                     "frequency": self.hparams.scheduler.lightning_args.frequency,
                                     "name": "learning_rate"
                                     }
                    }
        return {"optimizer": optimizer}
    
    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        duration_loss, prior_loss, diff_loss = self(
            x=x, 
            x_lengths = x_lengths, 
            y = y, 
            y_lengths = y_lengths, 
            spks = spks, 
            out_size = self.out_size
            )
        
        return {"duration_loss": duration_loss, 
                "prior_loss":prior_loss, 
                "diff_loss":diff_loss}
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]

    ## def training_step(self, batch: Any, batch_index: int):
    def shared_step(self, batch, mode):
        # mode: "train" or "val"
        loss_dict = self.get_losses(batch)

        batch_size = batch["x"].shape[0]

        self.log("step", 
                 float(self.global_step),
                 batch_size=batch_size,
                 on_step = True, 
                 on_epoch = True, 
                 logger = True, 
                 sync_dist = True)
        
        self.log(f"sub_loss/{mode}_duration_loss", 
                 loss_dict["duration_loss"], 
                 batch_size=batch_size,
                 on_step = True, 
                 on_epoch = True, 
                 logger = True, 
                 sync_dist = True)
        
        self.log(f"sub_loss/{mode}_prior_loss", 
                 loss_dict["prior_loss"], 
                 batch_size=batch_size,
                 on_step = True, 
                 on_epoch = True, 
                 logger = True, 
                 sync_dist = True)
        
        self.log(f"sub_loss/{mode}_diff_loss", 
                 loss_dict["diff_loss"], 
                 batch_size=batch_size,
                 on_step = True, 
                 on_epoch = True, 
                 logger = True, 
                 sync_dist = True)

        total_loss = sum(loss_dict.values())
        
        self.log(f"loss/{mode}", 
                 total_loss, 
                 batch_size=batch_size,
                 on_step = True, 
                 on_epoch = True, 
                 logger = True, 
                 prog_bar=True, 
                 sync_dist = True)

        if mode == "train":
            return {"loss": total_loss, "log":loss_dict}
        else:
            return total_loss
    
    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, mode = "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, mode = "val")

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting Original Samples")
                for i in range(2):
                    y  = one_batch["y"][i].unsqueeze(0).to(self.device)

                    # this is for other loggers:
                    # self.logger.experiment.add_image(
                    #     f"original/{i}",
                    #     plot_tensor(y.squeeze().cpu()),
                    #     self.current_epoch,
                    #     dataformats="HWC")
                    
                    # this is for wandb:
                    # https://docs.wandb.ai/guides/integrations/lightning
                    self.logger.experiment.log({f"ORIGIN_SAMPLE": [wandb.Image(plot_tensor(y.squeeze().cpu(), add_wandb_img = True))]})
                        
            log.debug("Synthesizing ...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps = 10, spks = spks)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]

                # this is for other loggers:
                # y_enc
                # self.logger.experiment.add_image(f"Generated_ENC/{i}", plot_tensor(y_enc.squeeze().cpu()), self.current_epoch, dataformats = "HWC")
                # y_dec
                # self.logger.experiment.add_image(f"Generated_DEC/{i}",  plot_tensor(y_dec.squeeze().cpu()), self.current_epoch, dataformats = "HWC")
                # alignments
                # self.logger.experiment.add_image(f"Alignment/{i}", plot_tensor(attn.squeeze().cpu()), self.current_epoch, dataformats = "HWC")

                # this is for wandb:
                # https://docs.wandb.ai/guides/integrations/lightning
                # y_enc
                self.logger.experiment.log({f"GEN_ENC_SAMPLE": [wandb.Image(plot_tensor(y_enc.squeeze().cpu(), add_wandb_img = True))]}) # caption = f"Generated_ENC/{i} @ Epoch {self.current_epoch}")]})
                # y_dec
                self.logger.experiment.log({f"GEN_DEC_SAMPLE": [wandb.Image(plot_tensor(y_dec.squeeze().cpu(), add_wandb_img = True))]}) # caption = f"Generated_DEC/{i} @ Epoch {self.current_epoch}")]})
                # attn
                self.logger.experiment.log({f"ALIGN_SAMPLE": [wandb.Image(plot_tensor(attn.squeeze().cpu(), add_wandb_img = True))]}) # caption = f"Alignment/{i} @ Epoch {self.current_epoch}")]})

    ## https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html
    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})


