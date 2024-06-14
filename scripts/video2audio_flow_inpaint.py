import argparse, os, sys, glob
import pathlib
directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
import random, math, librosa
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
from pathlib import Path
from tqdm import tqdm
def load_model_from_config(config, ckpt = None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]

        print(f'---------------------------epoch : {pl_sd["epoch"]}, global step: {pl_sd["global_step"]//1e3}k---------------------------')

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_rate",
        type=int,
        default="16000",
        help="sample rate of wav"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="length of wav"
    )
    parser.add_argument(
        "--test-dataset",
        default="vggsound",
        help="test which dataset: vggsound/landscape/fsd50k"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2audio-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=25,
        help="number of ddim sampling steps",
    )


    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default="",
    )


    return parser.parse_args()


def get_tail_mask(spec_truncate, gt_mel, gt_video_feat, fps, sr, truncate, hop_len, device):
    # apply mask
    masked_spec = int(spec_truncate * 0.5)  # 16帧的倍数，最多mask 50%
    masked_truncate = int(masked_spec * hop_len)
    masked_frame = int(fps * masked_truncate / sr)

    start_masked_idx = truncate * 0.25
    start_masked_frame = int(fps * start_masked_idx / sr)
    start_masked_spec = int(start_masked_idx / hop_len)

    spec = gt_mel.copy()
    spec[:, start_masked_spec:start_masked_spec + masked_spec] = torch.zeros((80, masked_spec))
    gt_video_feat[start_masked_frame:start_masked_frame + masked_frame, :] = np.zeros((masked_frame, 512))

    spec = torch.from_numpy(spec).unsqueeze(0).to(device)
    gt_video_feat = torch.from_numpy(gt_video_feat).unsqueeze(0).to(device)
    return spec, gt_video_feat


def get_random_mask(spec_truncate, gt_mel, gt_video_feat, fps, sr, truncate, hop_len, device):
    # apply mask
    masked_spec = random.randint(1, int(spec_truncate * 0.5 // 16)) * 16  # 16帧的倍数，最多mask 50%
    masked_truncate = int(masked_spec * hop_len)
    masked_frame = int(fps * masked_truncate / sr)

    start_masked_idx = random.randint(0, truncate - masked_truncate - 1)
    start_masked_frame = int(fps * start_masked_idx / sr)
    start_masked_spec = int(start_masked_idx / hop_len)

    spec = gt_mel.copy()
    spec[:, start_masked_spec:start_masked_spec + masked_spec] = torch.zeros((80, masked_spec))
    gt_video_feat[start_masked_frame:start_masked_frame + masked_frame, :] = np.zeros((masked_frame, 512))

    spec = torch.from_numpy(spec).unsqueeze(0).to(device)
    gt_video_feat = torch.from_numpy(gt_video_feat).unsqueeze(0).to(device)
    return spec, gt_video_feat

def main():
    opt = parse_args()

    config = OmegaConf.load(opt.base)
    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = CFMSampler(model, opt.ddim_steps)

    os.makedirs(opt.outdir, exist_ok=True)
    vocoder = VocoderBigVGAN(config['lightning']['callbacks']['image_logger']['params']['vocoder_cfg']['params']['ckpt_vocoder'], device)

    if os.path.exists('/apdcephfs/share_1316500/nlphuang/data/video_to_audio/vggsound/split_txt'):
        root = '/apdcephfs'
    else:
        root = '/apdcephfs_intern'

    if opt.test_dataset == 'vggsound':
        split, data = f'{root}/share_1316500/nlphuang/data/video_to_audio/vggsound/split_txt', f'{root}/share_1316500/nlphuang/data/video_to_audio/vggsound/'
        dataset1_spec_dir = os.path.join(data, "mel_maa2", "npy")
        dataset1_feat_dir = os.path.join(data, "cavp")

        with open(os.path.join(split, 'vggsound_test.txt'), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, x) + "_mel.npy", data_list1))
            video_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz",     data_list1))      # feat

    elif opt.test_dataset == 'landscape':
        split, data = f'{root}/share_1316500/nlphuang/data/video_to_audio/landscape/split/', f'{root}/share_1316500/nlphuang/data/video_to_audio/landscape/'

        dataset1_spec_dir = os.path.join(data, "melnone16000", "landscape_wav")
        dataset1_feat_dir = os.path.join(data, "landscape_visual_feat")
        with open(os.path.join(split, 'test.txt'), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, 'test', x) + ".npy", data_list1))      # spec
            video_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, 'test', x.replace('_mel', '')) + ".npy", data_list1))      # feat

    elif opt.test_dataset == 'Aist':
        split, data = f'{root}/share_1316500/nlphuang/data/video_to_audio/aist/split/', f'{root}/share_1316500/nlphuang/data/video_to_audio/aist/'

        dataset1_spec_dir = os.path.join(data, "melnone16000", "AIST++_crop_wav")
        dataset1_feat_dir = os.path.join(data, "AIST++_crop_visual_feat")
        with open(os.path.join(split, 'test.txt'), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, 'test', x) + ".npy", data_list1))      # spec
            video_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, 'test', x.replace('_mel', '')) + ".npy", data_list1))

    else:
        raise NotImplementedError


    sr, duration, truncate, fps = opt.sample_rate, config['data']['params']['train']['params']['dataset_cfg']['duration']\
        , config['data']['params']['train']['params']['dataset_cfg']['truncate'], config['data']['params']['train']['params']['dataset_cfg']['fps']
    hop_len = config['data']['params']['train']['params']['dataset_cfg']['hop_len']
    truncate_frame = int(fps * truncate / sr)
    spec_truncate = int(truncate / hop_len)

    if opt.scale != 1:
        unconditional = np.load(f'{root}/share_1316500/nlphuang/data/video_to_audio/vggsound/cavp/empty_vid.npz')['feat'].astype(np.float32)
        feat_len = fps * duration
        if unconditional.shape[0] < feat_len:
            unconditional = np.tile(unconditional, (math.ceil(feat_len / unconditional.shape[0]), 1))
        unconditional = unconditional[:int(feat_len)]
        unconditional = torch.from_numpy(unconditional).unsqueeze(0).to(device)
        unconditional = unconditional[:, :truncate_frame]


    # deal with long sequence
    shape = None
    if opt.length is not None:
        shape = (1, model.mel_dim, opt.length)

        from ldm.modules.diffusionmodules.flag_large_dit_moe import VideoFlagLargeDiT
        ntk_factor = opt.length // config['model']['params']['mel_length']
        # if hasattr(model.model.diffusion_model, 'ntk_factor') and ntk_factor != model.model.diffusion_model.ntk_factor:
        print(f"override freqs_cis, ntk_factor {ntk_factor}, flush=True")
        model.model.diffusion_model.freqs_cis = VideoFlagLargeDiT.precompute_freqs_cis(
            config['model']['params']['unet_config']['params']['hidden_size'] //
            config['model']['params']['unet_config']['params']['num_heads'],
            config['model']['params']['unet_config']['params']['max_len'],
            ntk_factor=ntk_factor
        )

    for i, (spec_path, video_feat_path) in enumerate(zip(spec_list1, video_list1)):
        name = Path(video_feat_path).stem

        if os.path.exists(os.path.join(opt.outdir, name + f'_0_gt.wav')):
            print(f'skip {name}')
            continue

        # waveform Features:
        try:
            spec_raw = np.load(spec_path).astype(np.float32)                    # channel: 1
        except:
            print(f"corrupted mel: {spec_path}", flush=True)
            spec_raw = np.zeros((80, 625), dtype=np.float32) # [C, T]

        try:
            video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
        except:
            video_feat = np.load(video_feat_path).astype(np.float32)

        spec_len = sr * duration / hop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]

        feat_len = fps * duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]

        window_num = video_feat.shape[0] // truncate_frame

        gt_mel_list, mel_list, masked_mel_list = [], [], []  # [sample_list1, sample_list2, sample_list3 ....]
        for i in tqdm(range(window_num), desc="Window:"):
            start, end = i * truncate_frame, (i + 1) * truncate_frame
            spec_start = int(start / fps * sr / hop_len)

            gt_video_feat = video_feat[start:end]
            gt_mel = spec_raw[:, spec_start: spec_start + spec_truncate]

            # apply mask
            # spec, gt_video_feat = get_random_mask(spec_truncate, gt_mel, gt_video_feat, fps, sr, truncate, hop_len, device)
            spec, gt_video_feat = get_tail_mask(spec_truncate, gt_mel, gt_video_feat, fps, sr, truncate, hop_len, device)


            # start sampling
            encoder_posterior = model.encode_first_stage(spec)
            z_spec = model.get_first_stage_encoding(encoder_posterior).detach()
            c = model.get_learned_conditioning({'mix_video_feat': gt_video_feat, 'mix_spec': z_spec})

            if opt.scale == 1: # w/o cfg
                sample, _ = sampler.sample(c, 1, timesteps=opt.ddim_steps, shape=shape)
            else:  # cfg
                uc = model.get_learned_conditioning({'mix_video_feat': unconditional, 'mix_spec': z_spec})
                sample, _ = sampler.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps, shape=shape)

            x_samples_ddim = model.decode_first_stage(sample)
            mel_list.append(x_samples_ddim)
            masked_mel_list.append(spec)
            gt_mel_list.append(torch.from_numpy(gt_mel).unsqueeze(0))

        if len(mel_list) > 0:
           syn_mel = np.concatenate([mel.cpu() for mel in mel_list], 1)
        if len(gt_mel_list) > 0:
           gt_mel = np.concatenate([mel for mel in gt_mel_list], 1)
        if len(gt_mel_list) > 0:
           masked_mel = np.concatenate([mel.cpu() for mel in masked_mel_list], 1)

        for idx, (spec, x_samples_ddim, spec_masked) in enumerate(zip(gt_mel, syn_mel, masked_mel)):
            wav = vocoder.vocode(spec)
            wav_path = os.path.join(opt.outdir, name + f'_{idx}_gt.wav')
            soundfile.write(wav_path, wav, opt.sample_rate)

            wav = vocoder.vocode(spec_masked)
            wav_path = os.path.join(opt.outdir, name + f'_{idx}_mask.wav')
            soundfile.write(wav_path, wav, opt.sample_rate)

            ddim_wav = vocoder.vocode(x_samples_ddim)
            wav_path = os.path.join(opt.outdir, name + f'_{idx}.wav')
            soundfile.write(wav_path, ddim_wav, opt.sample_rate)

    print(f"Your samples are ready and waiting four you here: \n{opt.outdir} \nEnjoy.")

if __name__ == "__main__":
    main()

