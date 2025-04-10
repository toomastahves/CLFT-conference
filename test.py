#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader

from utils.helpers import get_model_path, get_all_models

from tools.tester import Tester
from tools.dataset import Dataset

# Create results directory if it doesn't exist
def ensure_results_directory(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Created results directory: {results_path}")

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

test_data_files = [
    'test_day_fair.txt',
    'test_night_fair.txt',
    'test_day_rain.txt',
    'test_night_rain.txt'
]

if test_mode == '':
    print('Testing single model')
    model_path = get_model_path(config)
    tester = Tester(config, model_path)
    
    # Create a single results file for this model
    model_name = model_path.split('/')[-1]
    result_file_path = config['Log']['logdir'] + '/results/' + model_name + '.json'
    ensure_results_directory(config['Log']['logdir'] + '/results/')

    test_data_path = config['CLI']['path']
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
            tester.test_clft(test_dataloader, file, result_file_path)
        elif backbone == 'clfcn':
            tester.test_clfcn(test_dataloader, file, result_file_path)
    
    print('Testing is completed')

if test_mode == 'all':
    print('Testing all models in the folder')
    models = get_all_models(config)
    test_data_path = config['CLI']['path']
    
    for model_path in models:
        tester = Tester(config, model_path)
        
        # Create a single results file for this model
        model_name = model_path.split('/')[-1]
        result_file_path = config['Log']['logdir'] + '/results/' + model_name + '.json'
        ensure_results_directory(config['Log']['logdir'] + '/results/')
        
        for file in test_data_files:
            data_file_path = test_data_path + file
            print(f"Testing model {model_path} with data file {data_file_path}")

            test_data = Dataset(config, 'test', data_file_path)

            test_dataloader = DataLoader(test_data,
                                        batch_size=config['General']['batch_size'],
                                        shuffle=False,
                                        pin_memory=True,
                                        drop_last=True)

            if backbone == 'clft':
                tester.test_clft(test_dataloader, file, result_file_path)
            elif backbone == 'clfcn':
                tester.test_clfcn(test_dataloader, file, result_file_path)

    print('Testing is completed')
