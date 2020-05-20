import pickle
import shutil
import time
from pprint import pprint
from pathlib import Path

import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import Net
from parser import parse_args, parse_config
from dataset import Dataset


if __name__ == "__main__":
    args = parse_args()
    args = parse_config(args)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(exist_ok=True)

    if args.resume_checkpoint == 0:
        shutil.copy(args.cfg, args.model_dir / 'experiment_settings.cfg')
    else:
        shutil.copy(args.cfg, args.model_dir / 'experiment_settings_resume.cfg')

    if args.log_file.exists():
        args.log_file.unlink()
    logger.add(args.log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

    pprint(vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    model = Net(args)
    wandb_logger = WandbLogger(name='test',project='snips')
    trainer = Trainer(gpus=1, max_epochs=10, logger=wandb_logger) #num_sanity_val_steps=0 max_epochs=10 early_stop_callback=True
    trainer.fit(model)
    #trainer.test()

