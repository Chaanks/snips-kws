from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils import data
from sklearn import preprocessing

import numpy as np
from loguru import logger
from tqdm import tqdm

from utils import load_mfcc, load_json


UTT_LENGTH = 1000
WINDOW_SIZE = 50 # 25 a tester


class Dataset(data.Dataset):
    'Characterizes a dataset for Pytorch'
    def __init__(self, ds_path, mfcc_path):
        'Initialization'
        self.ds_path = ds_path
        self.mfcc_path = Path(mfcc_path / ds_path.stem).with_suffix('.p')
        self.ds_data = load_json(self.ds_path)
        self.utt2mfcc = load_mfcc(self.mfcc_path)
        self.mfcc2utt = {}
        logger.info('[{}/{}] training examples loaded from {}'.format(len(self.utt2mfcc), len(self.ds_data), self.ds_path.name))

        self.features, self.targets = self.prepare_data()


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.features)

    def __getitem__(self, idx):
            'Generates one sample of data'
            t = torch.Tensor([self.targets[idx]])
            return self.features[idx], t, idx

    def prepare_data(self):
        features = []
        targets = []
        mfcc_id = 0
        for utt in self.ds_data:
            if utt['id'] in self.utt2mfcc:
                #mfcc = self.utt2mfcc[utt['id']]
                mfcc = torch.from_numpy(preprocessing.scale(self.utt2mfcc[utt['id']], axis=0)).float()
            else:
                logger.warning('Missing mfcc for utt {}'.format(utt))

            for chunk in mfcc.split(UTT_LENGTH):
                padded = F.pad(input=chunk, pad=(0, 0, 0, UTT_LENGTH - chunk.shape[0]), mode='constant', value=0)
                windowed = padded.view(WINDOW_SIZE, -1)
                features.append(windowed)
                targets.append(utt['is_hotword'])
                self.mfcc2utt[mfcc_id] = utt['id']
                mfcc_id += 1
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