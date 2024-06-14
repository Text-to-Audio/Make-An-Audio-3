import csv
import os
import pickle
import sys

import numpy as np
import torch
import random
import math
import librosa

class audio_video_spec_fullset_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset1, feat_type='clip', transforms=None, sr=22050, duration=10, truncate=220000, fps=21.5, drop=0.0, fix_frames=False, hop_len=256):
        super().__init__()

        if split == "train":
            self.split = "Train"

        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        self.drop = drop
        print("Fix Frames: {}".format(self.fix_frames))
        print("Use Drop: {}".format(self.drop))

        # Dataset1: (VGGSound)
        assert dataset1.dataset_name == "VGGSound"
        
        # spec_dir: spectrogram path
        # feat_dir: CAVP feature path
        # video_dir: video path
        
        dataset1_spec_dir = os.path.join(dataset1.data_dir, "mel_maa2", "npy")
        dataset1_feat_dir = os.path.join(dataset1.data_dir, "cavp")
        dataset1_video_dir = os.path.join(dataset1.video_dir, "tmp_vid")
        
        split_txt_path = dataset1.split_txt_path
        with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))

            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, x) + "_mel.npy", data_list1))      # spec
            feat_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz",     data_list1))      # feat
            video_list1 = list(map(lambda x: os.path.join(dataset1_video_dir, x) + "_new_fps_21.5_truncate_0_10.0.mp4",   data_list1))      # video


        # Merge Data:
        self.data_list = data_list1 if self.split != "Test" else data_list1[:200]
        self.spec_list = spec_list1 if self.split != "Test" else spec_list1[:200]
        self.feat_list = feat_list1 if self.split != "Test" else feat_list1[:200]
        self.video_list = video_list1 if self.split != "Test" else video_list1[:200]

        assert len(self.data_list) == len(self.spec_list) == len(self.feat_list) == len(self.video_list)
        
        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.spec_list = [self.spec_list[i] for i in shuffle_idx]
        self.feat_list = [self.feat_list[i] for i in shuffle_idx]
        self.video_list = [self.video_list[i] for i in shuffle_idx]

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))



    def __len__(self):
        return len(self.data_list)
    

    def load_spec_and_feat(self, spec_path, video_feat_path):
        """Load audio spec and video feat"""
        try:
            spec_raw = np.load(spec_path).astype(np.float32)                    # channel: 1
        except:
            print(f"corrupted mel: {spec_path}", flush=True)
            spec_raw = np.zeros((80, 625), dtype=np.float32) # [C, T]

        p = np.random.uniform(0,1)
        if p > self.drop:
            try:
                video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
            except:
                print(f"corrupted video: {video_feat_path}", flush=True)
                video_feat = np.load(os.path.join(os.path.dirname(video_feat_path), 'empty_vid.npz'))['feat'].astype(np.float32)
        else:
            video_feat = np.load(os.path.join(os.path.dirname(video_feat_path), 'empty_vid.npz'))['feat'].astype(np.float32)

        spec_len = self.sr * self.duration / self.hop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]
        
        feat_len = self.fps * self.duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]
        return spec_raw, video_feat


    def mix_audio_and_feat(self, spec1=None, spec2=None, video_feat1=None, video_feat2=None, video_info_dict={}, mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)

            # Spec Start & Truncate:
            spec_start = int(start_idx / self.hop_len)
            spec_truncate = int(self.truncate / self.hop_len)

            spec1 = spec1[:, spec_start : spec_start + spec_truncate]
            video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]

            # info_dict:
            video_info_dict['video_time1'] = str(start_frame) + '_' + str(start_frame+truncate_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = ""
            return spec1, video_feat1, video_info_dict
        
        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len, total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len
            
            # concat spec:
            spec1, spec2 = spec1[:, spec_start1 : spec_end1], spec2[:, spec_start2 : spec_end2]
            concat_audio_spec = np.concatenate([spec1, spec2], axis=1)  

            # Concat Video Feat:
            start1_frame, truncate1_frame = int(self.fps * spec_start1 * self.hop_len / self.sr), int(self.fps * spec1_truncate_len * self.hop_len / self.sr)
            start2_frame, truncate2_frame = int(self.fps * spec_start2 * self.hop_len / self.sr), int(self.fps * self.truncate / self.sr) - truncate1_frame
            video_feat1, video_feat2 = video_feat1[start1_frame : start1_frame + truncate1_frame], video_feat2[start2_frame : start2_frame + truncate2_frame]
            concat_video_feat = np.concatenate([video_feat1, video_feat2])

            video_info_dict['video_time1'] = str(start1_frame) + '_' + str(start1_frame+truncate1_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = str(start2_frame) + '_' + str(start2_frame+truncate2_frame)
            return concat_audio_spec, concat_video_feat, video_info_dict 



    def __getitem__(self, idx):
        
        audio_name1 = self.data_list[idx]
        spec_npy_path1 = self.spec_list[idx]
        video_feat_path1 = self.feat_list[idx]
        video_path1 = self.video_list[idx]

        # select other video:
        flag = False
        if random.uniform(0, 1) < 0.5:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list)-1)
            audio_name2 = self.data_list[random_idx]
            spec_npy_path2 = self.spec_list[random_idx]
            video_feat_path2 = self.feat_list[random_idx]
            video_path2 = self.video_list[random_idx]

        # Load the Spec and Feat:
        spec1, video_feat1 = self.load_spec_and_feat(spec_npy_path1, video_feat_path1)

        if flag:
            spec2, video_feat2 = self.load_spec_and_feat(spec_npy_path2, video_feat_path2)
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': audio_name2, 'video_path1': video_path1, 'video_path2': video_path2}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1, spec2, video_feat1, video_feat2, video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': "", 'video_path1': video_path1, 'video_path2': ""}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1=spec1, video_feat1=video_feat1, video_info_dict=video_info_dict, mode='single')

        # print("mix spec shape:", mix_spec.shape)
        # print("mix video feat:", mix_video_feat.shape)
        data_dict = {}
        # data_dict['mix_spec'] = mix_spec[None].repeat(3, axis=0) # TODO：要把这里改掉，否则无法适应maa的autoencoder
        data_dict['mix_spec'] = mix_spec # (80, 512)
        data_dict['mix_video_feat'] = mix_video_feat # (32, 512)
        data_dict['mix_info_dict'] = mix_info

        return data_dict



class audio_video_spec_fullset_Dataset_Train(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class audio_video_spec_fullset_Dataset_Valid(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class audio_video_spec_fullset_Dataset_Test(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)



class audio_video_spec_fullset_Dataset_inpaint(audio_video_spec_fullset_Dataset):

    def __getitem__(self, idx):

        audio_name1 = self.data_list[idx]
        spec_npy_path1 = self.spec_list[idx]
        video_feat_path1 = self.feat_list[idx]
        video_path1 = self.video_list[idx]

        # Load the Spec and Feat:
        spec1, video_feat1 = self.load_spec_and_feat(spec_npy_path1, video_feat_path1)

        video_info_dict = {'audio_name1': audio_name1, 'audio_name2': "", 'video_path1': video_path1, 'video_path2': ""}
        mix_spec, mix_masked_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1=spec1, video_feat1=video_feat1, video_info_dict=video_info_dict)

        # print("mix spec shape:", mix_spec.shape)
        # print("mix video feat:", mix_video_feat.shape)
        data_dict = {}
        # data_dict['mix_spec'] = mix_spec[None].repeat(3, axis=0) # TODO：要把这里改掉，否则无法适应maa的autoencoder
        data_dict['mix_spec'] = mix_spec  # (80, 512)
        data_dict['hybrid_feat'] = {'mix_video_feat': mix_video_feat, 'mix_spec': mix_masked_spec}  # (32, 512)
        data_dict['mix_info_dict'] = mix_info

        return data_dict

    def mix_audio_and_feat(self, spec1=None, video_feat1=None, video_info_dict={}):
        """ Return Mix Spec and Mix video feat"""

        # spec1:
        if not self.fix_frames:
            start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
        else:
            start_idx = 0

        start_frame = int(self.fps * start_idx / self.sr)
        truncate_frame = int(self.fps * self.truncate / self.sr)

        # Spec Start & Truncate:
        spec_start = int(start_idx / self.hop_len)
        spec_truncate = int(self.truncate / self.hop_len)

        spec1 = spec1[:, spec_start: spec_start + spec_truncate]
        video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]

        # Start masking frames:
        masked_spec = random.randint(1, int(spec_truncate * 0.5 // 16)) * 16  # 16帧的倍数，最多mask 50%
        masked_truncate = int(masked_spec * self.hop_len)
        masked_frame = int(self.fps * masked_truncate / self.sr)

        start_masked_idx = random.randint(0, self.truncate - masked_truncate - 1)
        start_masked_frame = int(self.fps * start_masked_idx / self.sr)
        start_masked_spec = int(start_masked_idx / self.hop_len)

        masked_spec1 = np.zeros((80, spec_truncate)).astype(np.float32)
        masked_spec1[:] = spec1[:]
        masked_spec1[:, start_masked_spec:start_masked_spec+masked_spec] = np.zeros((80, masked_spec))
        video_feat1[start_masked_frame:start_masked_frame+masked_frame, :] = np.zeros((masked_frame, 512))
        # info_dict:
        video_info_dict['video_time1'] = str(start_frame) + '_' + str(start_frame + truncate_frame)  # Start frame, end frame
        video_info_dict['video_time2'] = ""
        return spec1, masked_spec1, video_feat1, video_info_dict



class audio_video_spec_fullset_Dataset_inpaint_Train(audio_video_spec_fullset_Dataset_inpaint):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class audio_video_spec_fullset_Dataset_inpaint_Valid(audio_video_spec_fullset_Dataset_inpaint):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class audio_video_spec_fullset_Dataset_inpaint_Test(audio_video_spec_fullset_Dataset_inpaint):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)



class audio_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset1, sr=22050, duration=10, truncate=220000, debug_num=False, fix_frames=False, hop_len=256):
        super().__init__()

        if split == "train":
            self.split = "Train"

        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        print("Fix Frames: {}".format(self.fix_frames))


        # Dataset1: (VGGSound)
        assert dataset1.dataset_name == "VGGSound"

        # spec_dir: spectrogram path

        # dataset1_spec_dir = os.path.join(dataset1.data_dir, "codec")
        dataset1_wav_dir = os.path.join(dataset1.wav_dir, "wav")

        split_txt_path = dataset1.split_txt_path
        with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            wav_list1 = list(map(lambda x: os.path.join(dataset1_wav_dir, x) + ".wav", data_list1))  # feat

        # Merge Data:
        self.data_list = data_list1
        self.wav_list = wav_list1

        assert len(self.data_list) == len(self.wav_list)

        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.wav_list = [self.wav_list[i] for i in shuffle_idx]

        if debug_num:
            self.data_list = self.data_list[:debug_num]
            self.wav_list = self.wav_list[:debug_num]

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)


    def load_spec_and_feat(self, wav_path):
        """Load audio spec and video feat"""
        try:
            wav_raw, sr = librosa.load(wav_path, sr=self.sr)                   # channel: 1
        except:
            print(f"corrupted wav: {wav_path}", flush=True)
            wav_raw = np.zeros((160000,), dtype=np.float32) # [T]

        wav_len = self.sr * self.duration
        if wav_raw.shape[0] < wav_len:
            wav_raw = np.tile(wav_raw, math.ceil(wav_len / wav_raw.shape[0]))
        wav_raw = wav_raw[:int(wav_len)]

        return wav_raw


    def mix_audio_and_feat(self, wav_raw1=None, video_info_dict={}, mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            wav_start = start_idx
            wav_truncate = self.truncate
            wav_raw1 = wav_raw1[wav_start: wav_start + wav_truncate]

            return wav_raw1, video_info_dict

        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len, total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len

            # concat spec:
            return video_info_dict


    def __getitem__(self, idx):

        audio_name1 = self.data_list[idx]
        wav_path1 = self.wav_list[idx]
        # select other video:
        flag = False
        if random.uniform(0, 1) < -1:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list)-1)
            audio_name2 = self.data_list[random_idx]
            spec_npy_path2 = self.spec_list[random_idx]
            wav_path2 = self.wav_list[random_idx]

        # Load the Spec and Feat:
        wav_raw1 = self.load_spec_and_feat(wav_path1)

        if flag:
            spec2, video_feat2 = self.load_spec_and_feat(spec_npy_path2, wav_path2)
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': audio_name2}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': ""}
            mix_wav, mix_info = self.mix_audio_and_feat(wav_raw1=wav_raw1, video_info_dict=video_info_dict, mode='single')

        data_dict = {}
        data_dict['mix_wav'] = mix_wav  # (131072,)
        data_dict['mix_info_dict'] = mix_info

        return data_dict


class audio_Dataset_Train(audio_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class audio_Dataset_Test(audio_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)

class audio_Dataset_Valid(audio_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)



class video_codec_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset1, sr=22050, duration=10, truncate=220000, fps=21.5, debug_num=False, fix_frames=False, hop_len=256):
        super().__init__()

        if split == "train":
            self.split = "Train"

        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.fps = fps
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        print("Fix Frames: {}".format(self.fix_frames))


        # Dataset1: (VGGSound)
        assert dataset1.dataset_name == "VGGSound"

        # spec_dir: spectrogram path

        # dataset1_spec_dir = os.path.join(dataset1.data_dir, "codec")
        dataset1_feat_dir = os.path.join(dataset1.data_dir, "cavp")
        dataset1_wav_dir = os.path.join(dataset1.wav_dir, "wav")

        split_txt_path = dataset1.split_txt_path
        with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            wav_list1 = list(map(lambda x: os.path.join(dataset1_wav_dir, x) + ".wav", data_list1))  # feat
            feat_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz", data_list1))  # feat

        # Merge Data:
        self.data_list = data_list1
        self.wav_list = wav_list1
        self.feat_list = feat_list1

        assert len(self.data_list) == len(self.wav_list)

        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.wav_list = [self.wav_list[i] for i in shuffle_idx]
        self.feat_list = [self.feat_list[i] for i in shuffle_idx]

        if debug_num:
            self.data_list = self.data_list[:debug_num]
            self.wav_list = self.wav_list[:debug_num]
            self.feat_list = self.feat_list[:debug_num]

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)


    def load_spec_and_feat(self, wav_path, video_feat_path):
        """Load audio spec and video feat"""
        try:
            wav_raw, sr = librosa.load(wav_path, sr=self.sr)                   # channel: 1
        except:
            print(f"corrupted wav: {wav_path}", flush=True)
            wav_raw = np.zeros((160000,), dtype=np.float32) # [T]

        try:
            video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
        except:
            print(f"corrupted video: {video_feat_path}", flush=True)
            video_feat = np.load(os.path.join(os.path.dirname(video_feat_path), 'empty_vid.npz'))['feat'].astype(np.float32)

        wav_len = self.sr * self.duration
        if wav_raw.shape[0] < wav_len:
            wav_raw = np.tile(wav_raw, math.ceil(wav_len / wav_raw.shape[0]))
        wav_raw = wav_raw[:int(wav_len)]

        feat_len = self.fps * self.duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]

        return wav_raw, video_feat


    def mix_audio_and_feat(self, wav_raw1=None, video_feat1=None, video_info_dict={}, mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            wav_start = start_idx
            wav_truncate = self.truncate
            wav_raw1 = wav_raw1[wav_start: wav_start + wav_truncate]

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)
            video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]

            # info_dict:
            video_info_dict['video_time1'] = str(start_frame) + '_' + str(start_frame+truncate_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = ""

            return wav_raw1, video_feat1, video_info_dict

        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len, total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len

            # concat spec:
            return video_info_dict


    def __getitem__(self, idx):

        audio_name1 = self.data_list[idx]
        wav_path1 = self.wav_list[idx]
        video_feat_path1 = self.feat_list[idx]
        # select other video:
        flag = False
        if random.uniform(0, 1) < -1:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list)-1)
            audio_name2 = self.data_list[random_idx]
            wav_path2 = self.wav_list[random_idx]
            video_feat_path2 = self.feat_list[random_idx]

        # Load the Spec and Feat:
        wav_raw1, video_feat1 = self.load_spec_and_feat(wav_path1, video_feat_path1)

        if flag:
            wav_raw2, video_feat2 = self.load_spec_and_feat(wav_path2, video_feat_path2)
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': audio_name2}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': ""}
            mix_wav, mix_video_feat, mix_info = self.mix_audio_and_feat(wav_raw1=wav_raw1, video_feat1=video_feat1, video_info_dict=video_info_dict, mode='single')

        data_dict = {}
        data_dict['mix_wav'] = mix_wav  # (131072,)
        data_dict['mix_video_feat'] = mix_video_feat # (32, 512)
        data_dict['mix_info_dict'] = mix_info

        return data_dict


class video_codec_Dataset_Train(video_codec_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class video_codec_Dataset_Test(video_codec_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)

class video_codec_Dataset_Valid(video_codec_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)


class audio_video_spec_fullset_Audioset_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset1, dataset2, sr=22050, duration=10, truncate=220000,
                 fps=21.5, drop=0.0, fix_frames=False, hop_len=256):
        super().__init__()

        if split == "train":
            self.split = "Train"

        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.sr = sr  # 22050
        self.duration = duration  # 10
        self.truncate = truncate  # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        self.drop = drop
        print("Fix Frames: {}".format(self.fix_frames))
        print("Use Drop: {}".format(self.drop))

        # Dataset1: (VGGSound)
        assert dataset1.dataset_name == "VGGSound"
        assert dataset2.dataset_name == "Audioset"

        # spec_dir: spectrogram path
        # feat_dir: CAVP feature path
        # video_dir: video path

        dataset1_spec_dir = os.path.join(dataset1.data_dir, "mel_maa2", "npy")
        dataset1_feat_dir = os.path.join(dataset1.data_dir, "cavp")
        split_txt_path = dataset1.split_txt_path
        with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))

            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, x) + "_mel.npy", data_list1))  # spec
            feat_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz", data_list1))  # feat

        if split == "train":
            dataset2_spec_dir = os.path.join(dataset2.data_dir, "mel")
            dataset2_feat_dir = os.path.join(dataset2.data_dir, "cavp_renamed")
            split_txt_path = dataset2.split_txt_path
            with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
                data_list2 = f.readlines()
                data_list2 = list(map(lambda x: x.strip(), data_list2))

                spec_list2 = list(map(lambda x: os.path.join(dataset2_spec_dir, f'Y{x}') + "_mel.npy", data_list2))  # spec
                feat_list2 = list(map(lambda x: os.path.join(dataset2_feat_dir, x) + ".npz", data_list2))  # feat

            data_list1 += data_list2
            spec_list1 += spec_list2
            feat_list1 += feat_list2

        # Merge Data:
        self.data_list = data_list1 if self.split != "Test" else data_list1[:200]
        self.spec_list = spec_list1 if self.split != "Test" else spec_list1[:200]
        self.feat_list = feat_list1 if self.split != "Test" else feat_list1[:200]

        assert len(self.data_list) == len(self.spec_list) == len(self.feat_list)

        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.spec_list = [self.spec_list[i] for i in shuffle_idx]
        self.feat_list = [self.feat_list[i] for i in shuffle_idx]

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))

        # self.check(self.spec_list)

    def __len__(self):
        return len(self.data_list)

    def check(self, feat_list):
        from tqdm import tqdm
        for spec_path in tqdm(feat_list):
            mel = np.load(spec_path).astype(np.float32)
            if mel.shape[0] != 80:
                import ipdb
                ipdb.set_trace()



    def load_spec_and_feat(self, spec_path, video_feat_path):
        """Load audio spec and video feat"""
        spec_raw = np.load(spec_path).astype(np.float32)  # channel: 1
        if spec_raw.shape[0] != 80:
            print(f"corrupted mel: {spec_path}", flush=True)
            spec_raw = np.zeros((80, 625), dtype=np.float32)  # [C, T]

        p = np.random.uniform(0, 1)
        if p > self.drop:
            try:
                video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
            except:
                print(f"corrupted video: {video_feat_path}", flush=True)
                video_feat = np.load(os.path.join(os.path.dirname(video_feat_path), 'empty_vid.npz'))['feat'].astype(np.float32)
        else:
            video_feat = np.load(os.path.join(os.path.dirname(video_feat_path), 'empty_vid.npz'))['feat'].astype(np.float32)

        spec_len = self.sr * self.duration / self.hop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]

        feat_len = self.fps * self.duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]
        return spec_raw, video_feat

    def mix_audio_and_feat(self, spec1=None, spec2=None, video_feat1=None, video_feat2=None, video_info_dict={},
                           mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)

            # Spec Start & Truncate:
            spec_start = int(start_idx / self.hop_len)
            spec_truncate = int(self.truncate / self.hop_len)

            spec1 = spec1[:, spec_start: spec_start + spec_truncate]
            video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]

            # info_dict:
            video_info_dict['video_time1'] = str(start_frame) + '_' + str(
                start_frame + truncate_frame)  # Start frame, end frame
            video_info_dict['video_time2'] = ""
            return spec1, video_feat1, video_info_dict

        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len,
                                                total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len

            # concat spec:
            spec1, spec2 = spec1[:, spec_start1: spec_end1], spec2[:, spec_start2: spec_end2]
            concat_audio_spec = np.concatenate([spec1, spec2], axis=1)

            # Concat Video Feat:
            start1_frame, truncate1_frame = int(self.fps * spec_start1 * self.hop_len / self.sr), int(
                self.fps * spec1_truncate_len * self.hop_len / self.sr)
            start2_frame, truncate2_frame = int(self.fps * spec_start2 * self.hop_len / self.sr), int(
                self.fps * self.truncate / self.sr) - truncate1_frame
            video_feat1, video_feat2 = video_feat1[start1_frame: start1_frame + truncate1_frame], video_feat2[
                                                                                                  start2_frame: start2_frame + truncate2_frame]
            concat_video_feat = np.concatenate([video_feat1, video_feat2])

            video_info_dict['video_time1'] = str(start1_frame) + '_' + str(
                start1_frame + truncate1_frame)  # Start frame, end frame
            video_info_dict['video_time2'] = str(start2_frame) + '_' + str(start2_frame + truncate2_frame)
            return concat_audio_spec, concat_video_feat, video_info_dict

    def __getitem__(self, idx):

        audio_name1 = self.data_list[idx]
        spec_npy_path1 = self.spec_list[idx]
        video_feat_path1 = self.feat_list[idx]

        # select other video:
        flag = False
        if random.uniform(0, 1) < -1:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list) - 1)
            audio_name2 = self.data_list[random_idx]
            spec_npy_path2 = self.spec_list[random_idx]
            video_feat_path2 = self.feat_list[random_idx]

        # Load the Spec and Feat:
        spec1, video_feat1 = self.load_spec_and_feat(spec_npy_path1, video_feat_path1)

        if flag:
            spec2, video_feat2 = self.load_spec_and_feat(spec_npy_path2, video_feat_path2)
            video_info_dict = {'audio_name1': audio_name1, 'audio_name2': audio_name2}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1, spec2, video_feat1, video_feat2, video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1': audio_name1, 'audio_name2': ""}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1=spec1, video_feat1=video_feat1, video_info_dict=video_info_dict, mode='single')

        # print("mix spec shape:", mix_spec.shape)
        # print("mix video feat:", mix_video_feat.shape)
        data_dict = {}
        data_dict['mix_spec'] = mix_spec  # (80, 512)
        data_dict['mix_video_feat'] = mix_video_feat  # (32, 512)
        data_dict['mix_info_dict'] = mix_info

        return data_dict


class audio_video_spec_fullset_Audioset_Train(audio_video_spec_fullset_Audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)


class audio_video_spec_fullset_Audioset_Valid(audio_video_spec_fullset_Audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)


class audio_video_spec_fullset_Audioset_Test(audio_video_spec_fullset_Audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)