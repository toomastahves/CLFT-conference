#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
    config = json.load(f)

files = glob.glob(config['Log']['logdir']+'progress_save/*.pth')
latest_file = max(files, key=os.path.getctime)
for file in files:
    if not file != latest_file:
        os.remove(file)
        print(f'Removed: {file}')