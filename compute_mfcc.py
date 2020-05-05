import shutil
import pickle
from pprint import pprint

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils import data

import numpy as np
from loguru import logger
from tqdm import tqdm

from utils import load_json
from parser import parse_args, parse_mfcc, get_mfcc_cfg


def make_mfcc(wav_path, mfcc_cfg):
    waveform, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == mfcc_cfg['sample_frequency']
    mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **mfcc_cfg)
    #print("Shape of mfcc: {}".format(mfcc.size()))
    return mfcc


def compute(ds_path, args):
    data = load_json(ds_path)
    mfcc_cfg = get_mfcc_cfg(args)
    logger.info('{} training examples loaded from {}'.format(len(data), ds_path.name))

    id2mfcc = {}
    cpt = 0
    for utt in tqdm(data):
        if utt['duration'] > 0:
            mfcc = make_mfcc(utt['audio_file_path'], mfcc_cfg)
            id2mfcc[utt['id']] = mfcc
            cpt += 1
        else:
            logger.warning('utt {} too short, duration of {} /s'.format(utt['id'], utt['duration']))

    logger.info('[{}/{}] mfcc computed'.format(cpt, len(data)))

    filepath = (args.mfcc_dir / ds_path.stem).with_suffix('.p')
    with open(filepath, 'wb') as f:
        pickle.dump(id2mfcc, f)


if __name__ == "__main__":
    args = parse_args()
    args = parse_mfcc(args)
    
    args.mfcc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.cfg, args.mfcc_dir / 'experiment_settings.cfg')

    if args.log_file.exists():
        args.log_file.unlink()
    logger.add(args.log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

    pprint(vars(args))

    compute(args.train_path, args)
    compute(args.dev_path, args)
    compute(args.test_path, args)