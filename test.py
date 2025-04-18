#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse
import os
import glob

from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset

parser = argparse.ArgumentParser(description='CLFT and CLFCN Testing')
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
    config = json.load(f)

np.random.seed(config['General']['seed'])
tester = Tester(config)

test_data_path = config['CLI']['path']
test_data_files = [
    'test_day_fair.txt',
    'test_night_fair.txt',
    'test_day_rain.txt',
    'test_night_rain.txt'
]

# Get all checkpoint files in the progress_save folder
checkpoint_pattern = os.path.join(config['Log']['logdir'], 'progress_save', '*.pth')
checkpoint_files = glob.glob(checkpoint_pattern)

if not checkpoint_files:
    print(f"No checkpoint files found in {checkpoint_pattern}")
    exit(1)

# Sort checkpoint files by epoch number
checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
print(f"Found {len(checkpoint_files)} checkpoint files")

for checkpoint_file in checkpoint_files:
    checkpoint_name = os.path.basename(checkpoint_file).split('.')[0]
    print(f"\nTesting checkpoint: {checkpoint_name}")
    tester.load_checkpoint(checkpoint_file)
    
    for file in test_data_files:
        path = test_data_path + file
        print(f"Testing with the path {path}")

        test_data = Dataset(config, 'test', path)

        test_dataloader = DataLoader(test_data,
                                    batch_size=config['General']['batch_size'],
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=True)

        # Include checkpoint name in the result file name
        result_file = config['Log']['logdir'] + '/results/' + 'result_' + file.replace('.txt', '') + '_' + checkpoint_name + '.csv'
        tester.test_clft(test_dataloader, config['CLI']['mode'], result_file)
        print(f'Testing for {file} with {checkpoint_name} is completed')
