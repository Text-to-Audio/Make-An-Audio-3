# Make-An-Audio 3: Transforming Text into Audio via Flow-based Large Diffusion Transformers

PyTorch Implementation of [Lumina-t2x](https://arxiv.org/abs/2405.05945)

We will provide our implementation and pretrained models as open source in this repository recently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.18474)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/AIGC-Audio/Lumina-Audio)
[![GitHub Stars](https://img.shields.io/github/stars/Text-to-Audio/Lumina-Audio?style=social)](https://github.com/Text-to-Audio/Lumina-Audio)

## Use pretrained model
We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://make-an-audio-2.github.io/) for audio samples.
## Quick Started
### Pretrained Models
Simply download the weights from [![Hugging Face](https://huggingface.co/Alpha-VLLM/Lumina-T2Music).  
- Text Encoder: [FLAN-T5-Large](https://huggingface.co/google/flan-t5-large)
- VAE: Make-An-Audio 2, finetuned from [Make an Audio](https://github.com/Text-to-Audio/Make-An-Audio)
- Decoder: [Vocoder](https://github.com/NVIDIA/BigVGAN)
- `Music` Checkpoints: [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-T2Music), `Audio' Checkpoints: [huggingface]()

### Generate audio/music from text
```
python3 scripts/txt2audio_for_2cap_flow.py 
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm1-cfg-LargeDiT3.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset audiocaps 
```

### Generate audio/music from audiocaps or musiccaps test dataset
- remember to relatively change `config["test_dataset]`
```
python3 scripts/txt2audio_for_2cap_flow.py 
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm1-cfg-LargeDiT3.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset testset
```

### Generate audio/music from video
```
python3 scripts/video2audio_flow.py 
--outdir output_dir -r  checkpoints_last.ckpt  -b configs/txt2audio-cfm1-cfg-LargeDiT3.yaml --scale 3.0 
--vocoder-ckpt useful_ckpts/bigvnat --test-dataset vggsound 
```

## Train
### Data preparation
- We can't provide the dataset download link for copyright issues. We provide the process code to generate melspec, count audio duration and generate structured caption.  
- Before training, we need to construct the dataset information into a tsv file, which includes name (id for each audio), dataset (which dataset the audio belongs to), audio_path (the path of .wav file),caption (the caption of the audio) ,mel_path (the processed melspec file path of each audio), duration (the duration of the audio). We provide a tsv file of audiocaps test set: audiocaps_test_struct.tsv as a sample.
- We provide a tsv file of the audiocaps test set: ./audiocaps_test_16000_struct.tsv as a sample.

### Generate the melspec file of audio
Assume you have already got a tsv file to link each caption to its audio_path, which mean the tsv_file have "name","audio_path","dataset" and "caption" columns in it.
To get the melspec of audio, run the following command, which will save mels in ./processed
```
python preprocess/mel_spec.py --tsv_path tmp.tsv --num_gpus 1 --max_duration 10
```

### Count audio duration
To count the duration of the audio and save duration information in tsv file, run the following command: 
```
python preprocess/add_duration.py --tsv_path tmp.tsv
```

### Generated structure caption from the original natural language caption
Firstly you need to get an authorization token in openai(https://openai.com/blog/openai-api), here is a tutorial(https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). Then replace your key of variable openai_key in preprocess/n2s_by_openai.py. Run the following command to add structed caption, the tsv file with structured caption will be saved into {tsv_file_name}_struct.tsv:
```
python preprocess/n2s_by_openai.py --tsv_path tmp.tsv
```

### Place Tsv files
After generated structure caption, put the tsv with structed caption to ./data/main_spec_dir . And put tsv files without structured caption to ./data/no_struct_dir

Modify the config data.params.main_spec_dir and  data.params.main_spec_dir.other_spec_dir_path respectively in config file configs/text2audio-ConcatDiT-ae1dnat_Skl20d2_struct2MLPanylen.yaml .

## Train variational autoencoder
Assume we have processed several datasets, and save the .tsv files in tsv_dir/*.tsv . Replace data.params.spec_dir_path with tsv_dir in the config file. Then we can train VAE with the following command. If you don't have 8 gpus in your machine, you can replace --gpus 0,1,...,gpu_nums
```
python main.py --base configs/research/autoencoder/autoencoder1d_kl20_natbig_r1_down2_disc2.yaml -t --gpus 0,1,2,3,4,5,6,7
```

## Train latent diffsuion
After trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```
python main.py --base configs/research/text2audio/text2audio-ConcatDiT-ae1dnat_Skl20d2_freezeFlananylen_drop.yaml -t  --gpus 0,1,2,3,4,5,6,7
```

## Evaluation
Please refer to [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio?tab=readme-ov-file#evaluation)


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