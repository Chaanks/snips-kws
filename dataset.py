import torch
import torchaudio
import torch.nn.functional as F
from torch.utils import data

import numpy as np
from loguru import logger
from tqdm import tqdm

from utils import load_json


UTT_LENGTH = 1000
WINDOW_SIZE = 10


def make_mfcc(wav_path, mfcc_cfg):
    waveform, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == mfcc_cfg['sample_frequency']
    mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **mfcc_cfg)
    #print("Shape of mfcc: {}".format(mfcc.size()))
    return mfcc

class Dataset(data.Dataset):
    'Characterizes a dataset for Pytorch'
    def __init__(self, ds_path, mfcc_cfg):
        'Initialization'
        self.ds_path = ds_path
        self.mfcc_cfg = mfcc_cfg
        self.ds_data = load_json(ds_path)
        logger.info('{} training examples loaded from {}'.format(len(self.ds_data), ds_path.name))

        self.features, self.targets = self.prepare_data()


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.features)

    def __getitem__(self, idx):
            'Generates one sample of data'
            # Select sample

    def prepare_data(self):
        features = []
        targets = []
        for idx, utt in tqdm(enumerate(self.ds_data)):
            mfcc = make_mfcc(utt['audio_file_path'], self.mfcc_cfg)
            for chunk in mfcc.split(UTT_LENGTH):
                padded = F.pad(input=chunk, pad=(0, 0, 0, UTT_LENGTH - chunk.shape[0]), mode='constant', value=0)
                features.append(padded.view(WINDOW_SIZE, -1))
                targets.append(utt['is_hotword'])
            #if mfcc.shape[0] > UTT_LENGTH:
                #print(self.ds_data[idx])
                #print(mfcc.shape)
                #a = input()
        return features, targets

    def target_one_hot(self, str_code):
        idx = torch.tensor(np.where(self.t_one_code == str_code)[0][0])
        #return torch.nn.functional.one_hot(idx, len(self.t_one_code))
        return idx

    def feature_one_hot(self, str_code):
        if str_code == '0':
            return torch.zeros((len(self.f_one_code,)), dtype=torch.int32)
        idx = torch.tensor(np.where(self.f_one_code == str_code)[0][0])
        return torch.nn.functional.one_hot(idx, len(self.f_one_code))