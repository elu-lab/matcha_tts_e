# 🍵 matcha_tts_e
TTS(= Text-To-Speech) model for studying and researching. This Repository is mainly based on [code](https://github.com/shivammehta25/Matcha-TTS/tree/main) and modified or added some codes. 

Languages
Dataset
wandb
Train
Synthesize

- 🔥[`Pytorch`](https://pytorch.org/), ⚡[`Lightning`](https://lightning.ai/docs/pytorch/stable/), 🐉[`hydra-core`](https://hydra.cc/docs/intro/)
- 🤗 `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/matcha_tts_e?nw=nwuserwako)
  - <details>
    <summary> dashboard screenshots </summary>
    <div>
    <img src="/readme_imgs/스크린샷 2024-05-25 오후 12.17.49.png" width="83%"></img>
    </div>
    </details>

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
We decided to use `monotonic_align` from this, and you can use like this:
```python
import monotonic_align
```

## Compute `mel_mean`, 'mel_std' of ljspeech dataset
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
data_statistics:  # Computed for ljspeech dataset # PYTHONPATH=. python matcha/utils/generate_data_statistics.py
  mel_mean: -5.517050   # {'mel_mean': -5.517050743103027, 'mel_std': 2.0643835067749023}
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
However, I failed to run for multi-gpu training. I really wanted to run in multi-gpu, I don't have enough time for this, When I tried, I got this message with training stoppped:

"child process with pid <number> terminated with code -11. forcefully terminating all other processes to avoid zombies 🧟"

## Synthesize
It will be continued.

## Reference
- 🍵 Paper: [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199)     
- :octocat: Github: [Official Code](https://github.com/shivammehta25/Matcha-TTS/tree/main)
- [monotonic_align 1.0.0](https://pypi.org/project/monotonic-align/)   
└ :octocat: [Github - moonsikpark/monotonic_align](https://github.com/moonsikpark/monotonic_align)     
└ :octocat: [mushanshanshan/monotonic_align](https://github.com/mushanshanshan/monotonic_align)     
