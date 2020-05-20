import time
from itertools import chain

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from sklearn import preprocessing 
from sklearn.metrics import f1_score

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from loguru import logger
import wandb

import parser
from dataset import Dataset
from utils import compute_score, weight_init, compute_precision_recall

class Net(pl.LightningModule):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        self.rnn = nn.GRU(260, 128, num_layers=2, dropout=0.5, bidirectional=False) #4000
        self.fc1 = nn.Linear(128 * 50, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(64, 1)
        #self.bn1 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        #print(x.shape)
        #print(x.view(x.shape[1], x.shape[0], -1).shape)
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

        logits = nn.Sigmoid()(logits)
        pred = torch.full(logits.shape, 0).cuda()
        pred[logits > 0.4] = 1

        correct_pred = pred.eq(y.view_as(pred)).sum().item()
        idpred = list(zip(idx.cpu(), pred.cpu(), y.cpu()))

        return {'val_loss': loss, 'correct': correct_pred, 'id_pred': idpred, 'logits': logits.cpu(), 'y_true': y.cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        all_pred = np.array(list(chain(*[x['id_pred'] for x in outputs])))
        logits = nn.Sigmoid()(torch.Tensor(list(chain(*[x['logits'] for x in outputs])))).reshape(-1, 1)
        y_true = np.array(list(chain(*[x['y_true'] for x in outputs])))

        score, acc = compute_score(self.ds_val, all_pred)
        #compute_precision_recall(y_true, logits)

        logits = np.hstack((np.zeros((logits.shape[0], 1)), logits))
        for l in logits:
            l[0] = 1 - l[1]

        logger.info(score)
        
        f1 = f1_score(y_true, all_pred[:, 1], labels=np.unique(all_pred))
        # Precision Recall
        self.logger.log_metrics({'pr': wandb.plots.precision_recall(y_true, logits, ["no_target", "target"])})
        logs = {'eval_loss': avg_loss, 'val_acc': acc, 'f1_score': f1}

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