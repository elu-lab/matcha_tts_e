# 🍵 matcha_tts_e
This repo is mainly based on :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main) and some codes are modified. The purpose of this repository is to study and study 🍵 [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199).

- 🔥[`Pytorch`](https://pytorch.org/), ⚡[`Lightning`](https://lightning.ai/docs/pytorch/stable/), 🐉🐲🐲[`hydra-core`](https://hydra.cc/docs/intro/)
- 🤗 **[`wandb`](https://kr.wandb.ai/)** Click 👉 [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/matcha_tts_e?nw=nwuserwako)
    <details>
    <summary>🦋 trim-butterfly-16 (<a href="https://wandb.ai/wako/matcha_tts_e/runs/77nc0bme?nw=nwuserwako">dashboard</a>) </summary>
    <div>
    - Batch Size: 16<br>
    - GPU: NVIDIA GeForce RTX 4080 <br>
    - GPU_COUNT: 1<br>
      <p>
        <img src="readme_imgs/스크린샷 2024-05-28 오전 9.30.14.png" alt="1" style="width:45%;"/>
        <img src="readme_imgs/스크린샷 2024-05-28 오전 9.30.31.png" alt="2" style="width:44%;"/>
     </p>
    </div>
    </details>

## Trying to code simpler
While studying :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main), I modified some codes to make it simpler.
- 🤗 Logger: **[`wandb`](https://kr.wandb.ai/)** (More comfortable and easy access)
- :fire: [`[Pytorch-Hub]NVIDIA/HiFi-GAN`](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/): used as a vocoder.
- MAS: :octocat: [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align) 👇

## MAS(=Monotonic Alignment Search) Installation
This is not included in [`requirements.txt`](https://github.com/elu-lab/matcha_tts_e/blob/main/requirements.txt). You can install MAS(Monotonic_Alignment_Search) with a following command below:     


:octocat: [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align)
```shell
pip install git+https://github.com/resemble-ai/monotonic_align.git
```
you can use like this:
```python
import monotonic_align
```

## Dataset: [**LJSpeech**](https://keithito.com/LJ-Speech-Dataset/)
  - `Language`: English :us:
  - `Speaker`: Single Speaker
  - `sample_rate`: 22.05kHz
    
## Compute `mel_mean`, `mel_std` of ljspeech dataset
Let's assume we are training with LJ Speech
1. Download the dataset from [here](https://keithito.com/LJ-Speech-Dataset/), extract it to your own data dir (In my case: `data/LJSpeech/ljs/LJSpeech-1.1`), and prepare the file lists to point to the extracted data like for item 5 in the setup of the [NVIDIA Tacotron 2 repo](https://github.com/NVIDIA/tacotron2#setup).
2. Go to `configs/data/ljspeech.yaml` and change
```yaml
train_filelist_path: data/filelists/ljs_audio_text_train_filelist.txt
valid_filelist_path: data/filelists/ljs_audio_text_val_filelist.txt
```
3. Generate normalisation statistics with the yaml file of dataset configuration
```shell
PYTHONPATH=. python matcha/utils/generate_data_statistics.py
```
4. Update these values in `configs/data/ljspeech.yaml` under `data_statistics` key.
```yaml
data_statistics:  # Computed for ljspeech dataset 
  mel_mean: -5.5170512199401855
  mel_std: 2.0643811225891113
```
Now you got ready to `train`!

## Train
First, you should log-in wandb with your token key in CLI. 
```
wandb login --relogin '<your-wandb-api-token>'
```
And you can run training with one of these commands:
```shell
PYTHONPATH=. python matcha/train.py experiment=ljspeech
```
```shell
# If you run training on a cetain gpu_id:
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python matcha/train.py experiment=ljspeech
```
Also, you can run for multi-gpu training:
```shell
# If you run multi-gpu training:
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python matcha/train.py experiment=ljspeech trainer.devices=[0,1]
```

## Synthesize
These codes are run and the example-speeches are synthesized in my vscode environment. I moved this Jupyter-Notebook file to Colab to share the synthesized example-speeches. Here is the [`Colab Notebook`](https://colab.research.google.com/drive/1JrwHDXrgcarZ7bxBAEP-cgBp6Yf_Ris4?usp=sharing).     
**Synthesize_Examples.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JrwHDXrgcarZ7bxBAEP-cgBp6Yf_Ris4?usp=sharing)           
<img src="/readme_imgs/스크린샷 2024-05-28 오전 8.52.42.png" width="67%"></img>
- You can refer to the code for synthesis: [`matcha/utils/synthesize_utils.py`](https://github.com/elu-lab/matcha_tts_e/blob/main/matcha/utils/synthesize_utils.py)
- `CLI Arguments`: Will be Updated!

## Reference
- 🍵 Paper: [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199)     
└ :octocat: Github: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main) 
- MAS(Monotonic Alignment Search)   
└ :octocat: [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align)
- 🔥 [`Pytorch`](https://pytorch.org/)
- ⚡ [`Lightning`](https://lightning.ai/docs/pytorch/stable/)
- 🐉🐲🐲 [`hydra-core`](https://hydra.cc/docs/intro/)
