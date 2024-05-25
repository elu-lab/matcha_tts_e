# 🍵 matcha_tts_e
This repo is mainly based on :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main) and some codes are modified. The purpose of this repository is to study and study 🍵 [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199).

- 🔥[`Pytorch`](https://pytorch.org/), ⚡[`Lightning`](https://lightning.ai/docs/pytorch/stable/), 🐉🐲🐲[`hydra-core`](https://hydra.cc/docs/intro/)
- 🤗 `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/matcha_tts_e?nw=nwuserwako) :point_left: Click 
  - <details>
    <summary> dashboard screenshots </summary>
    <div>
    <img src="/readme_imgs/스크린샷 2024-05-25 오후 12.17.49.png" width="83%"></img>
    </div>
    </details>

## Trying to code Simpler.
While studying :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main), I modified some codes to make it simpler.
- 🤗 Logger: `wandb` (More comfortable and easy access)
- :fire: [`[Pytorch-Hub]NVIDIA/HiFi-GAN`](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/): used as a vocoder.

## Limitations:
Encountered some unexpected errors when trying to run for training. I managed to solve the problem, but I failed to run for multi-gpu training. Instead of multi-gpu training, I could run for a single-gpu training. I also wanted multi-gpu training, but I don't have enough time for this. When I tried this at first, I got this message with training stoppped:

> *"child process with pid <number> terminated with code -11. forcefully terminating all other processes to avoid zombies 🧟"*

I'll write the CLI-command for multi-gpu training in the `Train` section.

## Dataset
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
  - `Language`: English :us:
  - `Speaker`: Single Speaker
  - `sample_rate`: 22.05kHz

## monotonic_align Installation
: you can install MAS(Monotonic_Alignment_Search) with one of following commands below:     
:octocat: [moonsikpark/monotonic_align](https://github.com/moonsikpark/monotonic_align)
```shell
pip install monotonic-align
```

:octocat: [mushanshanshan/monotonic_align](https://github.com/mushanshanshan/monotonic_align)
```shell
pip install git+https://github.com/mushanshanshan/monotonic_align.git
```
No matter what you choose to use, you can use like this:
```python
import monotonic_align
```

## Compute `mel_mean`, `mel_std` of ljspeech dataset
Let's assume we are training with LJ Speech
1. Download the dataset from [here](https://keithito.com/LJ-Speech-Dataset/), extract it to `data/LJSpeech-1.1`, and prepare the file lists to point to the extracted data like for item 5 in the setup of the [NVIDIA Tacotron 2 repo](https://github.com/NVIDIA/tacotron2#setup).
2. Go to `configs/data/ljspeech.yaml` and change
```yaml
train_filelist_path: data/filelists/ljs_audio_txt_train_filelist.txt
valid_filelist_path: data/filelists/ljs_audio_txt_val_filelist.txt
```
3. Generate normalisation statistics with the yaml file of dataset configuration
```shell
PYTHONPATH=. python matcha/utils/generate_data_statistics.py
```
4. Update these values in `configs/data/ljspeech.yaml` under `data_statistics` key.
```yaml
data_statistics:  # Computed for ljspeech dataset 
  mel_mean: -5.517050 
  mel_std: 2.064383
```
Now you got ready to `train`!

## Train
You can run training with one of these commands:
```shell
PYTHONPATH=. python matcha/train.py experiment=ljspeech
```
```shell
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python matcha/train.py experiment=ljspeech
```
Also, you can run for multi-gpu training:
```shell
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python matcha/train.py experiment=ljspeech trainer.devices=[0,1]
```

## Synthesize
It will be continued.

## Reference
- 🍵 Paper: [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199)     
- :octocat: Github: [Official Code](https://github.com/shivammehta25/Matcha-TTS/tree/main)
- MAS(Monotonic Alignment Search)   
└ :octocat: [Github - moonsikpark/monotonic_align](https://github.com/moonsikpark/monotonic_align)     
└ :octocat: [mushanshanshan/monotonic_align](https://github.com/mushanshanshan/monotonic_align)
- 🔥 [`Pytorch`](https://pytorch.org/)
- ⚡ [`Lightning`](https://lightning.ai/docs/pytorch/stable/)
- 🐉🐲🐲 [`hydra-core`](https://hydra.cc/docs/intro/)
