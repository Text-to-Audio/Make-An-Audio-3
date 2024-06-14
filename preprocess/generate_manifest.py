import glob

import numpy as np
from tqdm import tqdm
import torchaudio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import random
import os
import csv

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def generate():
    root = '/apdcephfs/share_1316500/nlphuang/data/text_to_audio/text_to_audio2/manifest/audioset-music/'
    MANIFEST_COLUMNS = ["name", "dataset", "ori_cap", "audio_path", "mel_path", "duration"]

    items = []
    with open(os.path.join(f'{root}/audioset_new.tsv'), encoding='utf-8') as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        items += [dict(e) for e in tqdm(reader)]
        assert len(items) > 0


    skip = 0
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for i, item in tqdm(enumerate(items)):
        mel_path = f'/apdcephfs//share_1316500/nlphuang/data/text_to_audio/text_to_audio2/music/mels/audioset/{Path(item["name"]).stem}_mel.npy'

        if not os.path.exists(mel_path):
            skip += 1
            continue

        manifest["name"].append(item['name'])
        manifest["dataset"].append(item['dataset'])
        manifest["ori_cap"].append(item['ori_cap'])
        manifest["duration"].append(item['duration'])
        manifest["audio_path"].append(item['audio_path'])
        manifest["mel_path"].append(mel_path)

    print(f"Writing manifest to {root}/audioset_new_intern.tsv..., skip: {skip}")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest),  f'{root}/audioset_new_intern.tsv')



if __name__ == '__main__':
    generate()