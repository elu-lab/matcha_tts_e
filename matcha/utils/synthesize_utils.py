import os
import sys

# import argparse
import datetime as dt
from pathlib import Path
# %matplotlib inline

import IPython.display as ipd
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn

# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import *
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HiFi-GAN
def get_hifigan(device):
    vocoder, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', trust_repo=True) ## Worked
    print(f"DownLoaded | NVIDIA's HiFi-GAN from torch hub | SR: 22050")
    vocoder.eval()
    vocoder = vocoder.to(device)
    denoiser = denoiser.to(device)

    return vocoder, vocoder_train_setup, denoiser

@torch.inference_mode()
def process_text(text, device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2']), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }

@torch.inference_mode()
def synthesise(text, 
               model,
               spks=None, 
               n_timesteps = 10, 
               length_scale=1.0, 
               temperature = 0.667,
               device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'].to(device), 
        text_processed['x_lengths'].to(device),
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_waveform(mel, vocoder, denoiser, denoising_strength = 0.00025):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=denoising_strength).cpu().squeeze()
    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


def synthesis_to_speech(texts, # = ["I really admire my professor most in the world"], 
                        model,
                        vocoder, # = None,
                        denoiser, 
                        denoising_strength = 0.00025,
                        spks=None, 
                        n_timesteps = 10,
                        length_scale = 1.0, 
                        temperature = 0.7,
                        save_wav=False, 
                        OUTPUT_FOLDER = './',
                        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(text, model = model, spks=spks, n_timesteps = n_timesteps, length_scale=length_scale, temperature = temperature) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output['waveform'] = to_waveform(output['mel'], vocoder, denoiser,  denoising_strength =  denoising_strength)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 22050 / (output['waveform'].shape[-1])

        ## Pretty print
        print(f"Input text: ")
        print(output['x_orig'])
        print(f"{'-' * 53}")
        print(f"Phonetised text: ")
        print(output['x_phones'])
        print(f"{'-' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output['rtf'])
        rtfs_w.append(rtf_w)

    ## Display the synthesised waveform
    ipd.display(ipd.Audio(output['waveform'], rate=22050))

    if save_wav:
        ## Save the generated waveform
        save_to_folder(i, output, OUTPUT_FOLDER)
    else:
        print("Not saving")
    
    return output, rtfs, rtfs_w

