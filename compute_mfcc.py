import torch
import torchaudio
import torch.nn.functional as F
from torch.utils import data

import numpy as np
from loguru import logger
from tqdm import tqdm

from utils import load_json
from parser import parse_args, parse_mfcc


def make_mfcc(wav_path, mfcc_cfg):
    waveform, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == mfcc_cfg['sample_frequency']
    mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **mfcc_cfg)
    #print("Shape of mfcc: {}".format(mfcc.size()))
    return mfcc


if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)

    mfcc_cfg = parser.get_mfcc_cfg(args)
    
    #ds_data = load_json(ds_path)
    #mfcc = make_mfcc(utt['audio_file_path'], mfcc_cfg)

