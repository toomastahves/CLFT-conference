#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from tqdm import tqdm

from utils.helpers import get_model_path
import utils.metrics as metrics
from clfcn.fusion_net import FusionNet
from clft.clft import CLFT


class Tester(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = config['CLI']['backbone']
        self.mode = config['CLI']['mode']
        self.path = config['CLI']['path']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if self.config['General']['model_specialization'] == 'large':
            self.nclasses = len(config['Dataset']['class_large_scale'])
        elif self.config['General']['model_specialization'] == 'small':
            self.nclasses = len(config['Dataset']['class_small_scale'])
        elif self.config['General']['model_specialization'] == 'all':
            self.nclasses = len(config['Dataset']['class_all_scale'])
        else:
            sys.exit("A specialization must be specified! (large or small or all)")

        if self.backbone == 'clfcn':
            self.model = FusionNet()
            print(f'Using backbone {self.backbone}')
            model_path = get_model_path(config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        elif self.backbone == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=self.nclasses,
                              model_timm=config['CLFT']['model_timm'],)
            print(f'Using backbone {self.backbone}')

            model_path = get_model_path(config)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)
        self.model.eval()

    def test_clft(self, test_dataloader, modal, result_file):
        print('CLFT Model Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                if self.config['General']['model_specialization'] == 'large':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_large_scale(self.nclasses, output_seg, anno)
                elif self.config['General']['model_specialization'] == 'small':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_small_scale(self.nclasses, output_seg, anno)
                elif self.config['General']['model_specialization'] == 'all':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_all_scale(self.nclasses, output_seg, anno)
                else:
                    sys.exit("A specialization must be specified! (large or small or all)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                progress_bar.set_description(f'IoU->{batch_IoU.cpu().numpy()}'
                                             f'Precision->{batch_precision.cpu().numpy()} '
                                             f'Recall->{batch_recall.cpu().numpy()}')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum
            print('-----------------------------------------')
            print(f'Testing result of {self.config["General"]["model_specialization"]} scale model,'
                  f'the modality is {self.mode}'
                  f'the subset is {self.path}')
            print(f'CUM_IoU->{cum_IoU.cpu().numpy()} '
                  f'CUM_Precision->{cum_precision.cpu().numpy()} '
                  f'CUM_Recall->{cum_recall.cpu().numpy()}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
            self.save_test_results(cum_IoU, cum_precision, cum_recall, result_file)

    def test_clfcn(self, test_dataloader, modal, result_file):
        print('CLFCN Model Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True).squeeze(1)

                outputs = self.model(batch['rgb'], batch['lidar'], modality)

                output = outputs[modality]
                annotation = batch['anno']
                if self.config['General']['model_specialization'] == 'large':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_large_scale(self.nclasses, output, annotation)
                elif self.config['General']['model_specialization'] == 'small':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_small_scale(self.nclasses, output, annotation)
                elif self.config['General']['model_specialization'] == 'all':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_all_scale(self.nclasses, output, annotation)
                else:
                    sys.exit("A specialization must be specified! (large or small or all)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                progress_bar.set_description(f'IoU->{batch_IoU.cpu().numpy()}'
                                             f'Precision->{batch_precision.cpu().numpy()} '
                                             f'Recall->{batch_recall.cpu().numpy()}')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            print('-----------------------------------------')
            print(f'Testing result of {self.config["General"]["model_specialization"]} scale model,'
                  f'the modality is {self.mode}'
                  f'the subset is {self.path}')
            print(f'CUM_IoU->{cum_IoU.cpu().numpy()} '
                  f'CUM_Precision->{cum_precision.cpu().numpy()} '
                  f'CUM_Recall->{cum_recall.cpu().numpy()}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
            self.save_test_results(cum_IoU, cum_precision, cum_recall, result_file)

    def save_test_results(self, cum_IoU, cum_precision, cum_recall, result_file):
        with open(result_file, 'a') as file:
            file.write('type,cum_IoU,cum_precision,cum_recall,average_precision\n')
            file.write(f'CUM_IoU->{cum_IoU.cpu().numpy()} \n')
            file.write(f'CUM_Precision->{cum_precision.cpu().numpy()} \n')
            file.write(f'CUM_Recall->{cum_recall.cpu().numpy()} \n')
