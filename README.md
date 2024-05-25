# 🍵 matcha_tts_e
TTS(= Text-To-Speech) model for studying and researching. This Repository is mainly based on [code](https://github.com/shivammehta25/Matcha-TTS/tree/main) and modified or added some codes. 

Languages
Dataset
wandb
Train
Synthesize

- ✍🏻🤗 `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/matcha_tts_e?nw=nwuserwako)
  - with ⚡ `Lightning`.
  - <details>
    <summary> dashboard screenshots </summary>
    <div>
    <img src="/imgs/스크린샷 2024-05-11 오후 10.18.04.png" width="83%"></img>
    <img src="/imgs/스크린샷 2024-05-11 오후 10.17.47.png" width="83%"></img>
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


## Reference
- 🍵 Paper: [Matcha-TTS: A fast TTS architecture with conditional flow matching](https://huggingface.co/papers/2309.03199)     
- :octocat: Github: [Official Code](https://github.com/shivammehta25/Matcha-TTS/tree/main)
- [monotonic_align 1.0.0](https://pypi.org/project/monotonic-align/)   
└ [Github - moonsikpark/monotonic_align](https://github.com/moonsikpark/monotonic_align)
