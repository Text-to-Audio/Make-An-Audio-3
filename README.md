# Make-An-Audio 3: Transforming Text into Audio via Flow-based Large Diffusion Transformers

PyTorch Implementation of [Lumina-t2x](https://arxiv.org/abs/2405.05945), [Lumina-Next](https://github.com/Alpha-VLLM/Lumina-T2X/blob/main/assets/lumina-next.pdf)

We will provide our implementation and pre-trained models as open-source in this repository recently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.18474)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3)
[![GitHub Stars](https://img.shields.io/github/stars/Text-to-Audio/Make-An-Audio-3?style=social)](https://github.com/Text-to-Audio/Make-An-Audio-3)


## News
- Oct, 2024: **[FashAudio](https://arxiv.org/abs/2410.12266)** released.
- Sept, 2024: **Lumina-Next](https://github.com/Text-to-Audio/Make-An-Audio-3)** accepted by NeurIPS'24.
- July, 2024: **[AudioLCM](https://arxiv.org/abs/2406.00356v1)** accepted by ACM-MM'24.
- June, 2024: **[Make-An-Audio 3 (Lumina-Next)](https://github.com/Text-to-Audio/Make-An-Audio-3)** released in Github and HuggingFace.
- May, 2024: **[AudioLCM](https://arxiv.org/abs/2406.00356v1)** released in Github and HuggingFace.

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

 Model     | Config                  | Pretraining Data |  Path  
|-----------|-------------------------|------------------|--------------------------------------------------------------------------------
| M (160M)  | txt2audio-cfm-cfg       | AudioCaption     |[TBD](https://huggingface.co/AIGC-Audio/Make-An-Audio-3/tree/main/text2audio/M)
| L (520M)  | /                       | AudioCaption     |[TBD]
| XL (750M) | txt2audio-cfm-cfg-XL    | AudioCaption     |[Here](https://huggingface.co/AIGC-Audio/Make-An-Audio-3/tree/main/text2audio/XL)
| XXL       | txt2audio-cfm-cfg-XXL   | AudioCaption     |[Here](https://huggingface.co/AIGC-Audio/Make-An-Audio-3/tree/main/text2audio/XXL)
| M (160M)  | txt2music-cfm-cfg       | Music            |[Here](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts/music_generation)
| L (520M)  | /                       | Music            |[TBD]
| XL (750M) | /                       | Music            |[TBD]
| 3B        | /                       | Music            |[TBD]
| M (160M)  | video2audio-cfm-cfg-moe | VGGSound         |[Here](https://huggingface.co/spaces/AIGC-Audio/Make-An-Audio-3/tree/main/useful_ckpts/video2audio)
### Generate audio/music from text
```
python3 scripts/txt2audio_for_2cap_flow.py  --prompt {TEXT}
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm-cfg.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat 
```
Add `--test-dataset structure` for text-to-audio generation

### Generate audio/music from audiocaps or musiccaps test dataset
- remember to alter `config["test_dataset"]`
```
python3 scripts/txt2audio_for_2cap_flow.py
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm-cfg.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset testset
```

### Generate audio from video
```
python3 scripts/video2audio_flow.py 
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/video2audio-cfm-cfg-moe.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset vggsound 
```

## Train flow-matching DiT
After trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```
python main.py --base configs/txt2audio-cfm-cfg.yaml -t  --gpus 0,1,2,3,4,5,6,7
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
```
@article{gao2024lumina-next,
  title={Lumina-Next: Making Lumina-T2X Stronger and Faster with Next-DiT},
  author={Zhuo, Le and Du, Ruoyi and Han, Xiao and Li, Yangguang and Liu, Dongyang and Huang, Rongjie and Liu, Wenze and others},
  journal={arXiv preprint arXiv:2406.18583},
  year={2024}
}
```

```
@article{huang2023make,
  title={Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models},
  author={Huang, Rongjie and Huang, Jiawei and Yang, Dongchao and Ren, Yi and Liu, Luping and Li, Mingze and Ye, Zhenhui and Liu, Jinglin and Yin, Xiang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.12661},
  year={2023}
}
```

```
@article{gao2024lumin-t2x,
  title={Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers},
  author={Gao, Peng and Zhuo, Le and Liu, Chris and and Du, Ruoyi and Luo, Xu and Qiu, Longtian and Zhang, Yuhang and others},
  journal={arXiv preprint arXiv:2405.05945},
  year={2024}
}

```

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
