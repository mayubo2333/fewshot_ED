import json
import jsonlines
import torch
import random
import numpy as np


def set_seed(args):
    if isinstance(args, int):
        random.seed(args)
        np.random.seed(args)
        torch.manual_seed(args)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        
def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def read_jsonlines(file_path):
    lines = list()
    with jsonlines.open(file_path) as reader:
        for line in reader:
            lines.append(line)
    return lines