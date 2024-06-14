import os.path

import torch
import torch.nn as nn
from functools import partial
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from importlib_resources import files
from ldm.modules.encoders.CLAP.utils import read_config_as_args
from ldm.modules.encoders.CLAP.clap import TextEncoder
from ldm.util import count_params
import numpy as np




class Video_Feat_Encoder_NoPosembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=40):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))

    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)        # B x 117 x C

        return x



class Video_Feat_Encoder_NoPosembed_inpaint(Video_Feat_Encoder_NoPosembed):
    """ Transform the video feat encoder"""

    def forward(self, x):
        # Revise the shape here:
        video, spec = x['mix_video_feat'], x['mix_spec']
        video = self.embedder(video)        # B x 117 x C

        return (video, spec)

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

