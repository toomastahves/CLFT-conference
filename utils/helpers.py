#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import json
import torch
import numpy as np


with open('config.json') as f:
    config = json.load(f)

if config['General']['model_specialization'] == 'small':
    all_classes = config['Dataset']['class_small_scale']
elif config['General']['model_specialization'] == 'large':
    all_classes = config['Dataset']['class_large_scale']
else:
    all_classes = config['Dataset']['class_all_scale']

class_values = [all_classes.index(cls.lower()) for cls in all_classes]


def creat_dir(config):
    logdir_rgb = config['Log']['logdir_rgb']
    logdir_lidar = config['Log']['logdir_lidar']
    logdir_fusion = config['Log']['logdir_fusion']
    if not os.path.exists(logdir_rgb):
        os.makedirs(logdir_rgb)
        print(f'Making log directory {logdir_rgb}...')
    if not os.path.exists(logdir_lidar):
        os.makedirs(logdir_lidar)
        print(f'Making log directory {logdir_lidar}...')
    if not os.path.exists(logdir_fusion):
        os.makedirs(logdir_fusion)
        print(f'Making log directory {logdir_fusion}...')

    if not os.path.exists(logdir_rgb + 'progress_save'):
        os.makedirs(logdir_rgb + 'progress_save')
    if not os.path.exists(logdir_lidar + 'progress_save'):
        os.makedirs(logdir_lidar + 'progress_save')
    if not os.path.exists(logdir_fusion + 'progress_save'):
        os.makedirs(logdir_fusion + 'progress_save')


def waymo_anno_class_relabel_large_scale(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->background
    :return annotation: 0->background+sign+cyclist+ignore, 1->vehicle, 2->pedestrian,
    """
    annotation = np.array(annotation)

    mask_ignore = annotation == 0
    mask_sign = annotation == 3
    mask_cyclist = annotation == 4
    mask_background = annotation == 5

    annotation[mask_sign] = 0
    annotation[mask_background] = 0
    annotation[mask_cyclist] = 0
    annotation[mask_ignore] = 0

    return torch.from_numpy(annotation).unsqueeze(0).long()  # [H,W]->[1,H,W]


def waymo_anno_class_relabel_small_scale(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->background
    :return annotation: 0->background+pedestrian+vehicle+ignore, 1-> cyclist 2->sign
    """
    annotation = np.array(annotation)

    mask_ignore = annotation == 0
    mask_vehicle = annotation == 1
    mask_pedestrian = annotation == 2
    mask_sign = annotation == 3
    mask_cyclist = annotation == 4
    mask_background = annotation == 5

    annotation[mask_background] = 0
    annotation[mask_cyclist] = 1
    annotation[mask_sign] = 2
    annotation[mask_ignore] = 0
    annotation[mask_vehicle] = 0
    annotation[mask_pedestrian] = 0

    return torch.from_numpy(annotation).unsqueeze(0).long()  # [H,W]->[1,H,W]


def waymo_anno_class_relabel_all_scale(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->background
    :return annotation: 0->background+ignore, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist,
    """
    annotation = np.array(annotation)
    mask_background = annotation == 5
    mask_ignore = annotation == 0

    annotation[mask_background] = 0
    annotation[mask_ignore] = 0

    return torch.from_numpy(annotation).unsqueeze(0).long()  # [H,W]->[1,H,W]


def waymo_anno_class_relabel_cross_scale(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    :parameter annotation: 0->background, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist
    :return annotation: 0->background, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist,
    """
    annotation = np.array(annotation)

    return torch.from_numpy(annotation).unsqueeze(0).long()  # [H,W]->[1,H,W]


def draw_test_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    model_specialization = config['General']['model_specialization']
    master_colors = {
        'background': (0, 0, 0),      # Black
        'vehicle': (0, 255, 0),       # Green
        'pedestrian': (0, 0, 255),    # Blue
        'sign': (255, 0, 0),          # Red
        'cyclist': (255, 255, 0)      # Yellow
    }

    if model_specialization == 'small':
        class_to_index = {
            'background': 0,
            'cyclist': 1,
            'sign': 2
        }
    elif model_specialization == 'large':
        class_to_index = {
            'background': 0,
            'vehicle': 1,
            'pedestrian': 2
        }
    else:  # 'all' or 'cross'
        class_to_index = {
            'background': 0,
            'vehicle': 1,
            'pedestrian': 2,
            'sign': 3,
            'cyclist': 4
        }

    for class_name, class_index in class_to_index.items():
        if class_name == 'background':
            continue  # Skip background class in visualization

        mask = (labels == class_index)
        if not np.any(mask):
            continue  # Skip if this class doesn't exist in the output

        red_map[mask] = master_colors[class_name][0]
        green_map[mask] = master_colors[class_name][1]
        blue_map[mask] = master_colors[class_name][2]

    # OpenCV uses BGR ordering, so we stack blue, green, red
    segmented_image = np.stack([blue_map, green_map, red_map], axis=2)
    return segmented_image


def image_overlay(image, segmented_image):
    result = image.copy()
    mask = (segmented_image[:,:,0] > 0) | (segmented_image[:,:,1] > 0) | (segmented_image[:,:,2] > 0)
    mask3d = np.stack([mask, mask, mask], axis=2)
    blended = np.where(mask3d, (segmented_image * 0.5 + image * 0.5).astype(np.uint8), result)
    return blended


def save_model_dict(config, epoch, model, modality, optimizer, save_check=False):
    sensor_modality = modality
    creat_dir(config)
    if save_check is False:
        if sensor_modality == 'rgb':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_rgb']+f"checkpoint_{epoch}.pth")
        elif sensor_modality == 'lidar':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_lidar']+f"checkpoint_{epoch}.pth")
        elif sensor_modality == 'cross_fusion':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_fusion']+f"checkpoint_{epoch}.pth")
    else:
        if sensor_modality == 'rgb':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_rgb']+'progress_save/'+f"checkpoint_{epoch}.pth")
        elif sensor_modality == 'lidar':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_lidar'] + 'progress_save/' + f"checkpoint_{epoch}.pth")
        elif sensor_modality == 'cross_fusion':
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       config['Log']['logdir_fusion'] + 'progress_save/' + f"checkpoint_{epoch}.pth")


def adjust_learning_rate_clft(config, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    epoch_max = config['General']['epochs']
    momentum = config['CLFT']['lr_momentum']
    # lr = config['General']['dpt_lr'] * (1-epoch/epoch_max)**0.9
    lr = config['CLFT']['clft_lr'] * (momentum ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_clfcn(config, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    epoch_max = config['General']['epochs']
    coefficient = config['CLFCN']['lr_coefficient']
    lr = config['CLFCN']['clfcn_lr'] * (1 - epoch/epoch_max) ** coefficient
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class EarlyStopping(object):
    def __init__(self, config):
        self.patience = config['General']['early_stop_patience']
        self.config = config
        self.min_param = None
        self.early_stop_trigger = False
        self.count = 0

    def __call__(self, valid_param, epoch, model, modality, optimizer):
        if self.min_param is None:
            self.min_param = valid_param
        elif valid_param >= self.min_param:
            self.count += 1
            print(f'Early Stopping Counter: {self.count} of {self.patience}')
            if self.count >= self.patience:
                self.early_stop_trigger = True
                print('Saving model for last epoch...')
                save_model_dict(self.config, epoch, model, modality, optimizer, True)
                print('Saving Model Complete')
                print('Early Stopping Triggered!')
        else:
            print(f'Valid loss decreases from {self.min_param:.4f} ' + f'to {valid_param:.4f}')
            self.min_param = valid_param
            save_model_dict(self.config, epoch, model, modality, optimizer)
            print('Saving Model...')
            self.count = 0
