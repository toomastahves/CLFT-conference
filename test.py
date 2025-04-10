#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import argparse
import numpy as np
from torch.utils.data import DataLoader

from utils.helpers import get_model_path, get_all_models

from tools.tester import Tester
from tools.dataset import Dataset


parser = argparse.ArgumentParser(description='CLFT Testing')
parser.add_argument('-c', '--config', type=str, required=False, help='The path of the config file')
args = parser.parse_args()
config_file = args.config
print("Config file path: ", args.config)

with open(config_file, 'r') as f:
    config = json.load(f)

print(config)
backbone = config['CLI']['backbone']
mode = config['CLI']['mode']

np.random.seed(config['General']['seed'])

test_mode = config['Test']['mode']

if test_mode == '':
    model_path = get_model_path(config)
    tester = Tester(config, model_path)

    test_data_path = config['CLI']['path']
    test_data_files = [
        'test_day_fair.txt',
        'test_night_fair.txt',
        'test_day_rain.txt',
        'test_night_rain.txt'
    ]
    for file in test_data_files:
        path = test_data_path + file
        print(f"Testing with the path {path}")

        test_data = Dataset(config, 'test', path)

        test_dataloader = DataLoader(test_data,
                                    batch_size=config['General']['batch_size'],
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=True)

        if backbone == 'clft':
            tester.test_clft(test_dataloader, file)
        elif backbone == 'clfcn':
            tester.test_clfcn(test_dataloader, file)
        print('Testing is completed')

if test_mode == 'all':
    print('Testing all models in the folder')
    models = get_all_models(config)
    print(models)

