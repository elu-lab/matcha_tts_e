# match_tts.py 
# Github: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/matcha_tts.py

import datetime as dt 
import math
import random

import torch

import monotonic_align as monotonic_align
# import matcha.utils.monotonic_align as monotonic_align
# pip install git+https://github.com/mushanshanshan/monotonic_align.git

from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import denormalize, duration_loss, fix_len_compatibility, generate_path, sequence_mask

log = utils.get_pylogger(__name__) ## 이건 왜 에러가 안 나지/이상한데....

# n_vocab = len(symbols); 178
# cleaners = ['english_cleaners2']
# add_blank = True
# spk_emb_dim =64
# n_spks = 1
# n_fft = 1024
# n_feats =  80
# sample_rate = 22050
# hop_length = 256
# win_length = 1024
# f_min = 0
# f_max = 8000
# out_size: null # Must be divisible by 4
# data_statistics = {"mel_mean": -5.536622, "mel_std": 2.116101}


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab, # 178 =  len(symbols)
        n_spks,  # 1
        spk_emb_dim, # 64
        n_feats,     # 80
        encoder,
        decoder,
        cfm,
        data_statistics, # data_statistics = {"mel_mean": -5.536622, "mel_std": 2.116101}
        out_size,        # out_size: null # Must be divisible by 4
        optimizer=None,
        scheduler=None,
        prior_loss=True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        # ==================================================================================== # 

        self.encoder = TextEncoder(
            encoder.encoder_type, # 'RoPE Encoder'
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        ## Encoder params
        # class Encoder_params():
        #     n_feats= n_feats
        #     n_channels =192
        #     filter_channels =  768
        #     filter_channels_dp =256
        #     n_heads= 2
        #     n_layers= 6
        #     kernel_size= 3
        #     p_dropout= 0.1
        #     spk_emb_dim= 64
        #     n_spks= 1
        #     prenet=True
        
        # encoder_params = Encoder_params()

        
        ## Duration Predictor params
        # class Duration_predictor_params():
        #     filter_channels_dp = 256 # ${model.encoder.encoder_params.filter_channels_dp}
        #     kernel_size= 3
        #     p_dropout = 0.1 # ${model.encoder.encoder_params.p_dropout}
            
        # duration_predictor_params= Duration_predictor_params()
        
        # ==================================================================================== # 

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder, # {'channels':[256, 256], 'dropout': 0.05, 'attention_head_dim': 64, 'n_blocks': 1, 'num_mid_blocks' : 2, 'num_heads': 2, 'act_fn': "snakebeta" }
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        
        # class CFM_parmas():
        #   name = 'CFM'
        #   solver = 'euler'
        #   sigma_min = 1e-4

        # cfm =CFM_parmas()

        self.update_data_statistics(data_statistics)

    
    # ==================================================================================== #
    
    ## Need Batches from DataLoader
    # from matcha.data.text_mel_datamodule import *

    # trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
    #     '/home/heiscold/matcha_tts/data/filelists/ljs_audio_text_train_filelist.txt', 
    #     # self.hparams.train_filelist_path,
    #     n_spks,    # self.hparams.n_spks,
    #     cleaners,  # self.hparams.cleaners,
    #     add_blank, # self.hparams.add_blank,
    #     n_fft,   # self.hparams.n_fft,
    #     n_feats, # self.hparams.n_feats,
    #     sample_rate, # self.hparams.sample_rate,
    #     hop_length,  # self.hparams.hop_length,
    #     win_length,  # self.hparams.win_length,
    #     f_min, # self.hparams.f_min,
    #     f_max, # self.hparams.f_max,
    #     data_statistics, # self.hparams.data_statistics,
    #     seed = 2024, # self.hparams.seed,
    #     )

    ## DataLoader
    # train_loader =  DataLoader(
    #     dataset = trainset, # dataset=self.trainset,
    #     batch_size = 4,     # batch_size=self.hparams.batch_size,
    #     #  num_workers=self.hparams.num_workers,
    #     #  pin_memory=self.hparams.pin_memory,
    #     shuffle= False,
    #     collate_fn=TextMelBatchCollate(n_spks), # collate_fn=TextMelBatchCollate(self.hparams.n_spks),
    #         )

    ## Sample Batch
    # sample_batch = next(iter(train_loader))
    # sample_batch.keys() # dict_keys(['x', 'x_lengths', 'y', 'y_lengths', 'spks'])

    # sample_batch['spks']: None

    ### forward(), synthesize()
    # sample_batch['x']: [bs, x_max_length] > [4, 293] 
    # sample_batch['x_lengths']: [bs] > [4]; [293, 251, 181, 221]

    ### forward()
    # sample_batch['y']: [bs, n_feats, 744] > [4, 80, 744]
    # sample_batch['y_lengths']: [bs] > [4]; [712, 742, 483, 628]
    
    
    @torch.inference_mode()
    def synthesise(self,
                   x, 
                   x_lengths, 
                   n_timesteps, 
                   temperature=1.0, 
                   spks=None, 
                   length_scale=1.0
                  ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
        """
        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        # x = sample_batch['x']
        # x_lengths = sample_batch['x_lengths']
        # spks = None

        # mu_x: [bs, n_feats, x_max_length] # [4, 80, 293] >> Encoder, proj_m Layer
        # logw: [bs, 1, x_max_length] # [4, 1, 293] >> from Duration Predictor
        # x_mask: [bs, 1, x_max_length]  # [4, 80, 293]


        w = torch.exp(logw) * x_mask # 
        # w: [bs, 1, x_max_length] # [4, 1, 293]
        # [[0.7519, 0.8960, 0.3594,  ..., 0.6103, 1.5431, 1.0207]],

        
        w_ceil = torch.ceil(w) * length_scale 
        # length_scale = 1.0
        # w_ceil: [bs, 1, x_max_length] # [4, 1, 293] # 소수점 올림
        # [[1., 1., 1.,  ..., 1., 2., 2.]],
        
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # y_lengths: [bs] >> [468, 386, 275, 333]
        # torch.sum(w_ceil[0, 0]): 468
        # torch.sum(w_ceil, [1, 2]) # [1, 2]: dimensions to reduce >> 468., 386., 275., 333.0]
        
        y_max_length = y_lengths.max() # 468
        # print(y_max_length) # tensor(468)
        y_max_length_ = fix_len_compatibility(y_max_length)
        # print(y_max_length_) # 468

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        # y_mask Shape: [bs, 1, y_max_length]; [4, 1, 468]
        # y_lengths: [bs] > [4]; [468, 386, 275, 333]
        
        ##### 'sequence_mask' function #####
        # ==================================================================================== #
        # def sequence_mask(length = y_lengths, max_length = y_max_length_ ):
        #     if max_length is None:
        #         max_length = length.max()
        #     x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        #     # x: [468]
        
        #     return x.unsqueeze(0) < length.unsqueeze(1)
        #     # x.unsqueeze(0): [1, 468]; [[0, 1, 2, ... 467]]
        #     # length(y_lengths).unsqueeze(1): [4, 1]; [[468], [386], [275], [333]]
        #     # ( x.unsqueeze(0) < length.unsqueeze(1) ).shape : [4, 468]
        #     #  x.unsqueeze(0) < length.unsqueeze(1) : [False, False, False,  ..., False, False, False], [False, False, False,  ...,  True,  True,  True],
        # ==================================================================================== #
        
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        # x_mask: [bs, 1, x_max_length]; [4, 1, 293] > x_mask.unsqueeze(-1): [bs, 1, x_max_length, 1]; [4, 1, 293, 1]
        # y_mask: [bs, 1, 468] > y_mask.unsqueeze(2): [bs, 1, 1, 468]; [4, 1, 1, 468]
        
        # attn_mask: [bs, 1, x_max_length, y_max_length]; [4, 1, 293, 468]
        
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        # w_ceil: [bs, 1, x_max_length]; [4, 1, 293]
        # w_ceil.squeeze(1): [bs, x_max_length]; [4, 293]
        # attn_mask.squeeze(1): [bs, x_max_length, y_max_length]; [4, 293, 468]

        # attn: [bs, x_max_length, y_max_length]; [4, 293, 468] > [bs, 1, x_max_length, y_max_length]; [4, 1, 293, 468]
        
        ##### 'generate_path' function #####
        # ==================================================================================== #
        # def generate_path(duration = w_ceil.squeeze(1), mask = attn_mask.squeeze(1) ):
        #     # duration = w_ceil.squeeze(1) # [4, 293]
        #     # mask = attn_mask.squeeze(1) # [4, 293, 468]
        #     device = duration.device
        
        #     b, t_x, t_y = mask.shape # # attn_mask.squeeze(1): [bs, x_max_length, y_max_length]; [4, 293, 468]
        
        #     cum_duration = torch.cumsum(duration, 1)
        #     # cum_duration.shape # [4, 293] > [  1.,   2.,   3.,  ..., 464., 466., 468.],
        
        #     path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

        #     cum_duration_flat = cum_duration.view(b * t_x)
        #     # cum_duration :[bs, 293] -> cum_duration_flat :[bs *293]; [1172]
        
        #     path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype) 
        #     # path: [bs * t_x, t_y] [1172, 468]
        
        #     path = path.view(b, t_x, t_y)
        #     # path: [bs * t_x, t_y] [1172, 468] -> [bs, t_x, t_y]; [4, 293, 468]
        
        #     path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        #     path = path * mask  # mask.shape: [bs, x_max_length, y_max_length]; [4, 293, 468]
        #     return path 
        #     # path: [bs, x_max_length, y_max_length]; [4, 293, 468]
        # ==================================================================================== #
        
        
        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        # attn.squeeze(1): [bs, x_max_length, y_max_length_]; [4, 293, 468]
        # attn.squeeze(1).transpose(1, 2): [bs, y_max_length_, x_max_length]; [4, 468, 293]

        # mu_x: [bs, n_feats, x_max_length]; [4, 80, 293]
        # mu_x.transpose(1, 2): [bs, x_max_length, n_feats]; [4, 293, 80]
        # >> mu_y: [bs, y_max_length, n_feats]; [4, 468, 80]
        
        mu_y = mu_y.transpose(1, 2) # mu_y: [bs, y_max_length, n_feats] > [bs, n_feats, y_max_length]
        encoder_outputs = mu_y[:, :, :y_max_length] # encoder_outputs: [bs, n_feats, y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        # decoder_outputs: [bs, n_feats, y_max_length_]; [4, 80, 468]

        # For RTF computation
        # t = dt.datetime.now() # 맨 위에 있음.
        t = (dt.datetime.now() - t).total_seconds() # 2.3e-05
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)  # 3.5062914823008848e-06
        
        return {
            "encoder_outputs": encoder_outputs, # [bs, n_feats, y_max_length]; [4, 80, 468]
            "decoder_outputs": decoder_outputs, # [bs, n_feats, y_max_length_]; [4, 80, 468]
            "attn": attn[:, :, :y_max_length],  # [bs, 1, x_max_length, y_max_length]; [4, 1, 293, 468] (if x_max_length <= y_max_length:)
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std), # [bs, n_feats, y_max_length_]; [4, 80, 468]
            "mel_lengths": y_lengths, #  [bs] >> [468, 386, 275, 333]
            "rtf": rtf, # 3.5062914823008848e-06
        }
    
    
    def forward(self,
                x,
                x_lengths,
                y,         # from `DataLaoder`
                y_lengths, # from `DataLaoder`
                spks = None,
                out_size = None,
                cond = None
                ):
                
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)
            
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        # x = sample_batch['x']
        # x_lengths = sample_batch['x_lengths']
        # spks = None

        # mu_x: [bs, n_feats, x_max_length] # [4, 80, 293] >> Encoder, proj_m Layer
        # logw: [bs, 1, x_max_length] # [4, 1, 293] >> from Duration Predictor
        # x_mask: [bs, 1, x_max_length]  # [4, 80, 293]

        y_max_length = y.shape[-1] # 744
        # y: [bs, n_feats, y_max_length]
        
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        # y_lengths: [bs]; [4] >> [712, 742, 483, 628]
        # y_max_length: 744
        # y_mask: [bs, 1, y_max_length]; [4, 1, 744]
        
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        # x_mask: [bs, 1, x_max_length]; [4, 1, 293] > x_mask.unsqueeze(-1): [bs, 1, x_max_length, 1]; [4, 1, 293, 1]
        # y_mask: [bs, 1, y_max_length]; [4, 1, 744]> y_mask.unsqueeze(2): [bs, 1, 1, y_max_length]; [4, 1, 1, 744]
        
        # attn_mask: [bs, 1, x_max_length, y_max_length]; [4, 1, 293, 744]
        
        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()
            # attn: [bs, x_max_length, y_max_length]; [4, 293, 744]
            # torch.unique(attn): tensor([0., 1.])
        
        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        # logw_: [bs, 1, x_max_length]; [4, 1, 293]
        # [[[0., 0., 0.,  ..., 0., 0., 0.]], [[0., 0., 0.,  ..., -0., -0., -0.]], ...
        
        dur_loss = duration_loss(logw, logw_, x_lengths)
        # dur_loss: tensor(38.1116, grad_fn=<DivBackward0>)
        
        
        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask
        
        
        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        # attn: [bs, x_max_length, y_max_length]; [4, 293, 744] -> attn.squeeze(1).transpose(1, 2): [bs, y_max_length, x_max_length]; [4, 744, 293]
        # mu_x: [bs, n_feats, x_max_length]; [4, 80, 293] -> mu_x.transpose(1, 2): [bs, x_max_length, n_feats]; [4, 293, 80]
        # mu_y: [bs, y_max_length, n_feats]; [4, 744, 80]
        mu_y = mu_y.transpose(1, 2)
        # mu_y: [bs, y_max_length, n_feats]; [4, 744, 80] -> [bs, n_feats, y_max_length]; [4, 80, 744]
        
        # Compute loss of the decoder
        # spks = None
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)
        # diff_loss: tensor(3.1869, grad_fn=<DivBackward0>)
        
        # prior_loss = True
        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            # prior_loss: tensor(979349.4375, grad_fn=<SumBackward0>)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            # prior_loss: tensor(4.7727, grad_fn=<DivBackward0>)
        else:
            prior_loss = 0
            
        return dur_loss, prior_loss, diff_loss
        # dur_loss: tensor(16.7201, grad_fn=<DivBackward0>)
        # prior_loss: tensor(4.7727, grad_fn=<DivBackward0>)
        # diff_loss: tensor(3.1869, grad_fn=<DivBackward0>))
