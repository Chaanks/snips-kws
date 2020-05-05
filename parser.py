import time
import json
import argparse
import configparser
from pathlib import Path

import numpy as np 


def parse_args():
    parser = argparse.ArgumentParser(description='Train SV model')
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--resume-checkpoint', type=int, default=0)
    args = parser.parse_args()
    assert args.cfg
    args.cfg = Path(args.cfg)
    assert args.cfg.is_file()
    args._start_time = time.ctime()
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_path = Path(config['Datasets'].get('train'))
    assert args.train_path
    args.dev_path= Path(config['Datasets'].get('dev'))
    args.test_path = Path(config['Datasets'].get('test')) 

    args.model_type = config['Model'].get('model_type', fallback='LSTM')
    assert args.model_type in ['LSTM']

    args.loss_type = config['Optim'].get('loss_type', fallback='BCEWithLogitsLoss')
    assert args.loss_type in ['BCEWithLogitsLoss']

    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.1)
    args.batch_size = config['Hyperparams'].getint('batch_size', fallback=255)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations = config['Hyperparams'].getint('num_iterations', fallback=50000)
    args.momentum = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.scheduler_steps = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.5)
    args.multi_gpu = config['Hyperparams'].getboolean('multi_gpu', fallback=False)
    args.log_interval = config['Hyperparams'].getfloat('log_interval', fallback=100)

    args.model_dir = Path(config['Outputs'].get('model_dir'))
    args.checkpoints_dir =  args.model_dir / 'checkpoints/'
    args.log_file = args.model_dir / 'train.log'
    args.results_pkl = args.model_dir / 'results.p'
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')

    return args

def parse_mfcc(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.train_path = Path(config['Datasets'].get('train'))
    args.dev_path= Path(config['Datasets'].get('dev'))
    args.test_path = Path(config['Datasets'].get('test')) 

    args.n_fft = config['MFCC'].getint('n_fft')
    args.frame_length = config['MFCC'].getint('frame_length')
    args.frame_shift = config['MFCC'].getint('frame_shift')
    args.channel = config['MFCC'].getint('channel')
    args.dither = config['MFCC'].getfloat('dither')
    args.window_type = config['MFCC'].get('window_type')
    args.sample_frequency = config['MFCC'].getint('sample_frequency')
    args.num_ceps = config['MFCC'].getint('num_ceps')
    args.num_mel_bins = config['MFCC'].getint('num_mel_bins')
    args.snip_edges = config['MFCC'].getboolean('snip_edges')

    args.mfcc_dir = Path(config['Outputs'].get('mfcc_dir'))
    args.log_file = args.mfcc_dir / 'compute_mfcc.log'

    return args

def get_mfcc_cfg(args):
    return {
        "channel": args.channel,
        "dither": args.dither,
        "window_type": args.window_type,
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "sample_frequency": args.sample_frequency,
        "num_ceps": args.num_ceps,
        "num_mel_bins": args.num_mel_bins,
        "snip_edges": args.snip_edges,
    }