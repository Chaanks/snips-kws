import json
import pickle

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


def compute_score(ds):
    result = {
        'true_true': 0,
        'true_false': 0,
        'false_true': 0,
        'false_false': 0,
    }
    
    for row in ds:
        idx, p, t = row
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