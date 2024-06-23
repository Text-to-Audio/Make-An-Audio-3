# Make-An-Audio 3: Transforming Text into Audio via Flow-based Large Diffusion Transformers

PyTorch Implementation of [Lumina-t2x](https://arxiv.org/abs/2405.05945), [Lumina-Next](https://github.com/Alpha-VLLM/Lumina-T2X/blob/main/assets/lumina-next.pdf)

We will provide our implementation and pretrained models as open source in this repository recently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.18474)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3)
[![GitHub Stars](https://img.shields.io/github/stars/Text-to-Audio/Make-An-Audio-3?style=social)](https://github.com/Text-to-Audio/Make-An-Audio-3)


## News
- June, 2024: **[Make-An-Audio-3 (Lumina-Next)](https://arxiv.org/abs/2405.05945)** released in [Github](https://github.com/Text-to-Audio/Make-An-Audio-3).

[//]: # (- May, 2024: **[Make-An-Audio-2]&#40;https://arxiv.org/abs/2207.06389&#41;** released in [Github]&#40;https://github.com/bytedance/Make-An-Audio-2&#41;.)
[//]: # (- August, 2023: **[Make-An-Audio]&#40;https://arxiv.org/abs/2301.12661&#41; &#40;ICML 2022&#41;** released in [Github]&#40;https://github.com/Text-to-Audio/Make-An-Audio&#41;. )

## Install dependencies

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

```bash
conda create -n Make_An_Audio_3 -y
conda activate Make_An_Audio_3
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
Install [nvidia apex](https://github.com/nvidia/apex) (optional)
```

## Quick Started
### Pretrained Models

Simply download the 500M weights from [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts)

 Model     | Task           | Pretraining Data |  Path  
|-----------|----------------|------------------|--------------------------------------------------------------------------------
| M (160M)  | Text-to-Audio  | AudioCaption     |[Here](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts/audio_generation)
| L (520M)  | Text-to-Audio  | AudioCaption     |[TBD]
| XL (750M) | Text-to-Audio  | AudioCaption     |[TBD]
| 3B        | Text-to-Audio  | AudioCaption     |[TBD]
| M (160M)  | Text-to-Music  | Music            |[Here](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts/music_generation)
| L (520M)  | Text-to-Music  | Music            |[TBD]
| XL (750M) | Text-to-Music  | Music            |[TBD]
| 3B        | Text-to-Music   | Music            |[TBD]
| M (160M)  | Video-to-Audio | VGGSound         |[Here](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts/video2audio)
### Generate audio/music from text
```
python3 scripts/txt2audio_for_2cap_flow.py  --prompt {TEXT}
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm1-cfg-LargeDiT3.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat 
```
Add `--test-dataset structure` for text-to-audio generation

### Generate audio/music from audiocaps or musiccaps test dataset
- remember to alter `config["test_dataset"]`
```
python3 scripts/txt2audio_for_2cap_flow.py
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm1-cfg-LargeDiT3.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset testset
```

### Generate audio from video
```
python3 scripts/video2audio_flow.py 
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/video2audio-cfm1-cfg-LargeDiT1-moe.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset vggsound 
```

## Train flow-matching DiT
After trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```
python main.py --base configs/txt2audio-cfm1-cfg-LargeDiT3.yaml -t  --gpus 0,1,2,3,4,5,6,7
```
## Others
For Data preparation, Training variational autoencoder, Evaluation, Please refer to [Make-An-Audio](https://github.com/liuhuadai/AudioLCM?tab=readme-ov-file#dataset-preparation).


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio),
[AudioLCM](https://github.com/Text-to-Audio/AudioLCM),
[CLAP](https://github.com/LAION-AI/CLAP),
as described in our code.


## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
```

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.