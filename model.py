import time
from itertools import chain

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from sklearn import preprocessing

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from loguru import logger

import parser
from dataset import Dataset
from utils import compute_score, weight_init

class Net(pl.LightningModule):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        self.rnn = nn.GRU(520, 128, num_layers=2, dropout=0.5, bidirectional=False) #4000
        #nn.init.xavier_uniform(self.lstm)
        self.fc1 = nn.Linear(128 * 50, 64)
        #nn.init.xavier_uniform(self.fc1.weight)
        #self.fc2 = nn.Linear(256, 64)
        #self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(64, 1)
        #nn.init.xavier_uniform(self.fc4.weight)
        self.dropout = nn.Dropout(p=0.2)
        #self.bn1 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        #print(x.shape)
        print(x.view(x.shape[1], x.shape[0], -1).shape)
        lstm_out, _ = self.rnn(x.view(x.shape[1], x.shape[0], -1))

        #print(lstm_out.shape)
        x = self.dropout(F.relu(self.fc1(lstm_out.view(lstm_out.shape[1], -1))))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)

        #prob = F.log_softmax(x, dim=1)
        return x

    def prepare_data(self):
        # stuff here is done once at the very beginning of training
        # before any distributed training starts

        # download stuff
        # save to disk
        # etc...
        self.apply(weight_init)

    def train_dataloader(self):
        # data transforms
        # dataset creation
        # return a DataLoader
        self.ds_train = Dataset(self.cfg.train_path, self.cfg.mfcc_path)
        train_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': True,
                        'num_workers': 6}

        return DataLoader(self.ds_train, **train_params)

    def val_dataloader(self):
        # can also return a list of val dataloaders
        self.ds_val = Dataset(self.cfg.dev_path, self.cfg.mfcc_path)
        val_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': False,
                        'num_workers': 6}

        return DataLoader(self.ds_val, **val_params)

    def test_dataloader(self):
        # can also return a list of test dataloaders
        self.ds_test =Dataset(self.cfg.test_path, self.cfg.mfcc_path)
        test_params = {'batch_size': self.cfg.batch_size,
                        'shuffle': False,
                        'num_workers': 6}

        return DataLoader(self.ds_test, **test_params)


    def configure_optimizers(self):
        #return optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)
        return optim.Adam(self.parameters(), lr=self.cfg.lr)#amsgrad=True

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)

        logs = {'train_loss': loss}
        #self.logger.log_metrics(logs)

        return {'loss': loss, 'log': logs}


    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)

        pred = torch.full(logits.shape, 0).cuda()
        pred[logits > 0] = 1

        correct_pred = pred.eq(y.view_as(pred)).sum().item()
        idpred = list(zip(idx.cpu(), pred.cpu(), y.cpu()))

        return {'val_loss': loss, 'correct': correct_pred, 'id_pred': idpred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        all_pred = list(chain(*[x['id_pred'] for x in outputs]))

        score, acc = compute_score(self.ds_val, all_pred)
        
        logger.info(score)
        logs = {'eval_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'log': logs}


    def test_step(self, batch, batch_idx):
        x, y, idx = batch
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs}