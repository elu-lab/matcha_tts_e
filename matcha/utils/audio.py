## Github[match.utils.audio.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/audio.py

import numpy as np
import torch
import torch.nn as nn

from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read


MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    ## scipy.io.wavefile.read
    sampling_rate, data = read(full_path)
    return data, sampling_rate

## numpy
def dynamic_range_compression(x, C=1, clip_val = 1e-5):
    return np.log(np.clip(x, a_min = clip_val, a_max = None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

## torch
def dynamic_range_compression_torch(x, C=1, clip_val = 1e-5):
    return torch.log(torch.clamp(x, min = clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C



def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_denormalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output



mel_basis = {}
hann_window = {}


def mel_spectrogram(y, 
                    n_fft, 
                    num_mels,
                    sampling_rate,
                    hop_length,
                    win_length,
                    fmin,
                    fmax,
                    center=False
                    ):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        # print(f"{str(fmax)}_{str(y.device)}") # 32768_cpu
        mel = librosa_mel_fn(sr=sampling_rate, 
                             n_fft=n_fft, 
                             n_mels=num_mels, 
                             fmin=fmin, 
                             fmax=fmax)
        # print(mel.shape) # [80, 513]
        # Dictionary
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), ## [1, 108462] --> [1, 1, 108462]
         (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), 
        mode="reflect"
    ) # [1, 1, 109230]

    y = y.squeeze(1) # [1, 1, 109230] -> [1, 109230]

    spec = torch.view_as_real(
        torch.stft(
            y, # [1, 19292]
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window[str(y.device)], # torch.hann_window(win_length).to(y.device)
            center=center, # False
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ) ## torch.stft --> torch.Size([1, 513, 423])
    ) ## torch.Size([1, 513, 423, 2])

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))  ## torch.Size([1, 513, 423])

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    # mel_basis[str(fmax) + "_" + str(y.device)]: [80, 513]
    # spec: torch.Size([1, 513, 423])
    # >> spece: torch.Size([1, 80, 423]
    spec = spectral_normalize_torch(spec)
    
    return spec
