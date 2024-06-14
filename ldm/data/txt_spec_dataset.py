import csv
import os
import pickle
import sys

import numpy as np
import torch
import random
import math
import librosa
import pandas as pd
from pathlib import Path
class audio_spec_join_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset_name, spec_crop_len, drop=0.0):
        super().__init__()

        if split == "train":
            self.split = "Train"

        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.spec_crop_len = spec_crop_len
        self.drop = drop

        print("Use Drop: {}".format(self.drop))

        self.init_text2audio(dataset_name)

        print('Split: {}  Total Sample Num: {}'.format(split, len(self.dataset)))

        if os.path.exists('/apdcephfs_intern/share_1316500/nlphuang/data/video_to_audio/vggsound/cavp/empty_vid.npz'):
            self.root = '/apdcephfs_intern'
        else:
            self.root = '/apdcephfs'


    def init_text2audio(self, dataset):

        with open(dataset) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]

        if self.split == 'Test':
            samples = samples[:100]

        self.dataset = samples
        print('text2audio dataset len:', len(self.dataset))

    def __len__(self):
        return len(self.dataset)
    
    def load_feat(self, spec_path):
        try:
            spec_raw = np.load(spec_path)  # mel spec [80, T]
        except:
            print(f'corrupted mel:{spec_path}', flush=True)
            spec_raw = np.zeros((80, self.spec_crop_len), dtype=np.float32) # [C, T]

        spec_len = self.spec_crop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]

        return spec_raw


    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        p = np.random.uniform(0, 1)
        if p > self.drop:
            caption = {"ori_caption": data['ori_cap'], "struct_caption": data['caption']}
        else:
            caption = {"ori_caption": "", "struct_caption": ""}

        mel_path = data['mel_path'].replace('/apdcephfs', '/apdcephfs_intern') if self.root == '/apdcephfs_intern' else data['mel_path']
        spec = self.load_feat(mel_path)

        data_dict['caption'] = caption
        data_dict['image'] = spec  # (80, 624)

        return data_dict


class spec_join_Dataset_Train(audio_spec_join_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class spec_join_Dataset_Valid(audio_spec_join_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class spec_join_Dataset_Test(audio_spec_join_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)



class audio_spec_join_audioset_Dataset(audio_spec_join_Dataset):

    # def __init__(self, split, dataset_name, root, spec_crop_len, drop=0.0):
    #     super().__init__(split, dataset_name, spec_crop_len, drop)
    #
    #     self.data_dir = root
        # MANIFEST_COLUMNS = ["name", "dataset", "ori_cap", "audio_path", "mel_path", "duration"]
        # manifest = {c: [] for c in MANIFEST_COLUMNS}
        # skip = 0
        # if self.split != 'Train': return
        # from preprocess.generate_manifest import save_df_to_tsv
        # from tqdm import tqdm
        # for idx in tqdm(range(len(self.dataset))):
        #     item = self.dataset[idx]
        #     mel_path = f'{self.data_dir}/{Path(item["name"])}_mel.npy'
        #     try:
        #         _ = np.load(mel_path)
        #     except:
        #         skip += 1
        #         continue
        #
        #     manifest["name"].append(item['name'])
        #     manifest["dataset"].append("audioset")
        #     manifest["ori_cap"].append(item['ori_cap'])
        #     manifest["duration"].append(item['audio_path'])
        #     manifest["audio_path"].append(item['duration'])
        #     manifest["mel_path"].append(mel_path)
        #
        # print(f"Writing manifest to {dataset_name.replace('audioset.tsv', 'audioset_new.tsv')}..., skip: {skip}")
        # save_df_to_tsv(pd.DataFrame.from_dict(manifest), f"{dataset_name.replace('audioset.tsv', 'audioset_new.tsv')}")


    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        p = np.random.uniform(0, 1)
        if p > self.drop:
            caption = data['ori_cap']
        else:
            caption = ""
        spec = self.load_feat(data['mel_path'])

        data_dict['caption'] = caption
        data_dict['image'] = spec  # (80, 624)

        return data_dict



class spec_join_Dataset_audioset_Train(audio_spec_join_audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class spec_join_Dataset_audioset_Valid(audio_spec_join_audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class spec_join_Dataset_audioset_Test(audio_spec_join_audioset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)
