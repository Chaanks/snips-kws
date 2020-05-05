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