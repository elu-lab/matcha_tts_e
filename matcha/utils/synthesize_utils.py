import os
import sys

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

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HiFi-GAN
def get_hifigan( device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ):
    vocoder, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', trust_repo=True) ## Worked
    print(f"DownLoaded | NVIDIA's HiFi-GAN from torch hub | SR: 22050")
    vocoder.eval()
    vocoder = vocoder.to(device)
    denoiser = denoiser.to(device)

    return vocoder, vocoder_train_setup, denoiser 

# vocoder, vocoder_train_setup, denoiser = get_hifigan(device)

# Synthesize Helper functions
@torch.inference_mode()
def process_text(text: str):
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
def synthesise(text, spks=None, n_timesteps = 10, length_scale=1.0, temperature = 0.667, device = device):
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
def to_waveform(mel, vocoder, denoising_strength = 0.00025):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()
    
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


# Synthesize-to-speech function
def synthesis_to_speech(texts = ["I really admire my professor most in the world"], 
                        spks=None, 
                        n_timesteps = 10,    # Number of ODE Solver steps
                        length_scale=1.0,    # Changes to the speaking rate
                        temperature = 0.667, # Sampling temperature
                        save_wav=False, 
                        OUTPUT_FOLDER = './',
                        device = device):
    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(text, spks=None, n_timesteps = 10, length_scale=1.0, temperature = 0.667) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output['waveform'] = to_waveform(output['mel'], vocoder)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w = t * 22050 / (output['waveform'].shape[-1])

        ## Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output['x_orig'])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output['x_phones'])
        print(f"{'*' * 53}")
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
