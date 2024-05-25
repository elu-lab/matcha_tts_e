## Github[flow_matching.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)



class BASECFM(nn.Module, ABC):
    def __init__(self, 
                 n_feats, 
                 cfm_params, 
                 n_spks = 1, 
                 spk_emb_dim = 128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver =  cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4
            
        self.estimator = None ## Decoder

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        
        ## Author's words: Storing this bcz he can later plot this by putting a debugger here and saving it to a file
        ##                  or in future might add like a return_all_steps flag

        ## Seemes `changed version`
        sol = []

        for step in range(1, len(t_span)):
            # self.estimator = Decoder
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            # Equation (1)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

        ### Before `changed`
        # sol = []
        # steps = 1
        # while steps <= len(t_span) - 1:
        #     dph_dt = self.estimator(x, mask, mu, t, spks, cond)
        #     ## self.estimator = decoder
        #     ## Equation 01

        #     x = x + dt * dphi_dt
        #     t = t + dt
        #     sol.append(x)
        #     if steps < len(t_span) -1:
        #         dt = t_span[steps + 1] - t
        #     steps += 1

        # return sol[-1]


    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature = 1.0, spks = None, cond = None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device = mu.device)
        return self.solve_euler(z, t_span = t_span, mu = mu, mask = mask, spks = spks, cond = cond)

    def compute_loss(self, x1, mask, mu, spks = None, cond = None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """

        b, _, t = mu.shape

        ## random timestep
        t = torch.rand([b, 1, 1], device = mu.device, dtype = mu.dtype)
        # sample noise p(x, 0)
        z = torch.randn_like(x1)

        # ϕOTt (x) = (1 − (1 − σmin)t)x0 + tx1
        y = (1 - (1 - self.sigma_min) * t ) * z + t * x1
        # uOTt (ϕOTt (x0)|x1) = x1 − (1 − σmin)x0
        u = x1 - (1 - self.sigma_min) * z

        ## Equation (4)
        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze(), spks,), u, reduction = "sum") / (torch.sum(mask) * u.shape[1])

        ## why return 'y' ???
        return loss, y



class CFM(BASECFM):
    def __init__(self, 
                 in_channels, 
                 out_channel, 
                 cfm_params, 
                 decoder_params, 
                 n_spks =1, 
                 spk_emb_dim = 64
                 ):
        super().__init__(
            n_feats = in_channels,
            cfm_params = cfm_params,
            n_spks = n_spks,
            spk_emb_dim = spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the Architecture of the esimator here
        self.estimator = Decoder(in_channels = in_channels,
                                 out_channels = out_channel,
                                 **decoder_params)

