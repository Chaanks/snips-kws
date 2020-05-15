import json
import pickle

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.init as init

def load_mfcc(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_2_col(path):
    data = {}
    with open(path) as fp:
        for line in fp:
            col = line.strip().split(' ')
            data[col[0]] = col[1]
    return data


def compute_score(ds, all_pred):
    result = {
        'true_true': 0,
        'true_false': 0,
        'false_true': 0,
        'false_false': 0,
    }

    #logger.add('debug.log', format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

    id2pred = {}
    for row in all_pred:
        idx, p, t = row[0].item(), row[1].item(), row[2].item()
        utt_id = ds.mfcc2utt[idx]
        if utt_id not in id2pred:
            id2pred[utt_id] = (p, t)
        else:
            if p == 1 and id2pred[utt_id][0] == 0:
                id2pred[utt_id] = (p, t)
        #logger.debug('idx: {} utt_id: {} pred: {} dict: {}'.format(idx, utt_id, p, id2pred[utt_id]))

    for k, v in id2pred.items():
        p, t = v
        if p == 1 and t == 1:
            result['true_true'] += 1
        elif p == 0 and t == 0:
            result['true_false'] += 1
        elif p == 1 and t == 0:
            result['false_true'] += 1
        elif p == 0 and t == 1:
            result['false_false'] += 1

    size = sum([x for x in result.values()])
    acc = 100. *  (result['true_true'] + result['true_false']) / size
    return result, acc

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
