# 🍵 matcha_tts_e
This repo is mainly based on :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main) and some codes are modified. The purpose of this repository is to study and study 🍵 [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199).

- 🔥[`Pytorch`](https://pytorch.org/), ⚡[`Lightning`](https://lightning.ai/docs/pytorch/stable/), 🐉🐲🐲[`hydra-core`](https://hydra.cc/docs/intro/)
- 🤗 **[`wandb`](https://kr.wandb.ai/)** Click 👉 [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/matcha_tts_e?nw=nwuserwako)


## Trying to code simpler
While studying :octocat: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main), I modified some codes to make it simpler.
- 🤗 Logger: **[`wandb`](https://kr.wandb.ai/)** (More comfortable and easy access)
- :fire: [`[Pytorch-Hub]NVIDIA/HiFi-GAN`](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/): used as a vocoder.
- **MAS:** :octocat: [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align) 👇
- **MemoryCleanupCallback Added!**
  
  ```python
  import gc
  import torch
  import lightning as L
  
    class MemoryCleanupCallback(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        def on_validation_epoch_end(self, trainer, pl_module):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
  ```

## Colab notebooks (Examples):
These codes are run and the example-speeches are synthesized in my vscode environment. I moved this Jupyter Notebook file to Colab to share the synthesized example-speeches below:    

- **Samples_trim_butterfly_16.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M_CgnwZCt1kYQhxohAR3beUUn40553lR?usp=sharing) | `BS`: `16` | `GPU`: `NVIDIA GeForce RTX 4080 (x1)`       
- **Samples_decent_meadow_46.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BV0F9dXfmAZg390zC68s7feXkugBK8nL?usp=sharing) | `BS`: `32` | `LR`: `2e-5` | `GPU`: `NVIDIA GeForce RTX 4080 (x1)`           
- :star: **Samples_wobbly_frog_53.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/193Sauiz-GdMIJslbH3I56bWqSmO_iXPF?usp=sharing) | `BS`: `16` | `Precision`: `bf16-mixed` | `GPU`: `NVIDIA GeForce RTX 4080 (x1)`                
- **Samples_wobbly_serenity_54.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/136jutbUw6sQVDPP4ccQlIjoyRRW_sD-R?usp=sharing) | `BS`: `32` | `Precision`: `bf16-mixed` | `GPU`: `NVIDIA GeForce RTX 4080 (x1)`               
- **Samples_jolly_frog_47.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UWypCHOsQQJF-HX3vToNc7zah6lMcPUg?usp=sharing) | `BS`: `32` | `LR`: `2e-5` | `GPU`: `NVIDIA GeForce RTX 4090 (x1)`      
- :star2: **Samples_eager_frost_50.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iy7v1CWAA0rUqoYN2nh5INVZ43OqX1j0?usp=sharing) | `BS`: `16` | `GPU`: `NVIDIA GeForce RTX 4090 (x1)`             
- :sparkles: **Samples_royal_grass_56.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OklqorBjE7X11XHkKuCofly-Nc8iuQTu?usp=sharing) | `BS`: `16` | `Precision`: `bf16-mixed` | `GPU`: `NVIDIA GeForce RTX 4090 (x1)`                    

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
These codes are run and the example-speeches are synthesized in my vscode environment. I moved this Jupyter-Notebook file to Colab to share the synthesized example-speeches.      

**Samples_wobbly_frog_53.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/193Sauiz-GdMIJslbH3I56bWqSmO_iXPF?usp=sharing)      

- you can check more samples **Colab notebooks (Examples)** above.
- You can refer to the code for synthesis: [`matcha/utils/synthesize_utils.py`](https://github.com/elu-lab/matcha_tts_e/blob/main/matcha/utils/synthesize_utils.py)
- This notebook is also on this github repo: [`notebooks/Synthesize_Examples.ipynb`](https://github.com/elu-lab/matcha_tts_e/blob/main/notebooks/Synthesize_Examples.ipynb)
- `CLI Arguments`: Will be Updated!

## Reference
- 🍵 Paper: [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199)     
└ :octocat: Github: [🍵 Matcha-TTS Official Github](https://github.com/shivammehta25/Matcha-TTS/tree/main) 
- MAS(Monotonic Alignment Search)   
└ :octocat: [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align)
- 🔥 [`Pytorch`](https://pytorch.org/)
- ⚡ [`Lightning`](https://lightning.ai/docs/pytorch/stable/)
- 🐉🐲🐲 [`hydra-core`](https://hydra.cc/docs/intro/)
