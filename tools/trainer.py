#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import utils.metrics as metrics
from clfcn.fusion_net import FusionNet
from clft.clft import CLFT
from utils.helpers import EarlyStopping
from utils.helpers import save_model_dict
from utils.helpers import adjust_learning_rate_clft
from utils.helpers import adjust_learning_rate_clfcn


writer = SummaryWriter()


class Trainer(object):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.finished_epochs = 0
        self.device = torch.device(self.config['General']['device']
                                   if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if self.config['General']['model_specialization'] == 'large':
            self.nclasses = len(config['Dataset']['class_large_scale'])
            weight_loss = torch.tensor(self.config['Dataset']['class_weight_large_scale'])
        elif self.config['General']['model_specialization'] == 'small':
            self.nclasses = len(config['Dataset']['class_small_scale'])
            weight_loss = torch.tensor(self.config['Dataset']['class_weight_small_scale'])
        elif self.config['General']['model_specialization'] == 'all':
            self.nclasses = len(config['Dataset']['class_all_scale'])
            weight_loss = torch.tensor(self.config['Dataset']['class_weight_all_scale'])
        elif self.config['General']['model_specialization'] == 'cross':
            self.nclasses = len(config['Dataset']['class_cross_scale'])
            weight_loss = torch.tensor(self.config['Dataset']['class_weight_cross_scale'])
        else:
            sys.exit("A specialization must be specified! (large or small or all or cross)")
        self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)

        if args.backbone == 'clfcn':
            self.model = FusionNet()
            print(f'Using backbone {args.backbone}')
            self.optimizer_clfcn = torch.optim.Adam(self.model.parameters(), lr=config['CLFCN']['clfcn_lr'])
        elif args.backbone == 'clft':
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
            print(f'Using backbone {args.backbone}')
            self.optimizer_clft = torch.optim.Adam(self.model.parameters(), lr=config['CLFT']['clft_lr'])
        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")
        self.model.to(self.device)

        if self.config['General']['resume_training'] is True:
            print('Resume training...')
            model_path = self.config['General']['resume_training_model_path']
            checkpoint = torch.load(model_path, map_location=self.device)

            if self.config['General']['reset_lr'] is True:
                print('Reset the epoch to 0')
                self.finished_epochs = 0
            else:
                self.finished_epochs = checkpoint['epoch']
                print( f"Finished epochs in previous training: {self.finished_epochs}")

            if self.config['General']['epochs'] <= self.finished_epochs:
                print('Current epochs amount is smaller than finished epochs!!!')
                print(f"Please setting the epochs bigger than {self.finished_epochs}")
                sys.exit()
            else:
                print('Loading trained model weights...')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print('Loading trained optimizer...')
                if args.backbone == 'clft':
                    self.optimizer_clft.load_state_dict(checkpoint['optimizer_state_dict'])
                elif args.backbone == 'clfcn':
                    self.optimizer_clfcn.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    sys.exit("A backbone must be specified! (clft or clfcn)")

        else:
            print('Training from the beginning')

    def train_clft(self, train_dataloader, valid_dataloader, modal):
        """
        The training of one epoch
        """
        epochs = self.config['General']['epochs']
        modality = modal
        early_stopping = EarlyStopping(self.config)
        self.model.train()
        for epoch in range(self.finished_epochs, epochs):
            lr = adjust_learning_rate_clft(self.config, self.optimizer_clft, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clft.zero_grad()

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
                elif self.config['General']['model_specialization'] == 'cross':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_cross_scale(self.nclasses, output_seg, anno)
                else:
                    sys.exit("A specialization must be specified! (large or small or all or cross)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])

                train_loss += loss.item()
                loss.backward()
                self.optimizer_clft.step()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')

            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print(f'Training IoU for Epoch: {train_epoch_IoU}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clft(valid_dataloader, modality)

            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss,
                                        'valid': valid_epoch_loss}, epoch)
            writer.close()

            early_stop_index = round(valid_epoch_loss, 4)
            early_stopping(early_stop_index, epoch, self.model, modality, self.optimizer_clft)
            save_epoch = self.config['General']['save_epoch']
            if (epoch + 1) % save_epoch == 0 and epoch > 0:
                print(f'Saving model for every {save_epoch} epochs...')
                save_model_dict(self.config, epoch, self.model, modality, self.optimizer_clft, True)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
        print('Training Complete')

    def validate_clft(self, valid_dataloader, modal):
        """
            The validation of one epoch
        """
        self.model.eval()
        print('Validating...')
        valid_loss = 0.0
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                output_seg = self.model(batch['rgb'], batch['lidar'], modal)
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
                elif self.config['General']['model_specialization'] == 'cross':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_cross_scale(self.nclasses, output_seg, anno)
                else:
                    sys.exit("A specialization must be specified! (large or small or all or cross)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])
                valid_loss += loss.item()
                progress_bar.set_description(f'valid fusion loss: {loss:.4f}')
        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        print(f'Validation IoU for Epoch: {valid_epoch_IoU}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i + 1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        return valid_epoch_loss, valid_epoch_IoU

    def train_clfcn(self, train_dataloader, valid_dataloader, modal):
        """
        The training of one epoch
        """
        epochs = self.config['General']['epochs']
        modality = modal
        early_stopping = EarlyStopping(self.config)
        self.model.train()
        for epoch in range(self.finished_epochs, epochs):
            lr = adjust_learning_rate_clfcn(self.config, self.optimizer_clfcn, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)
            for i, batch in enumerate(progress_bar):
                train_loss = 0.0
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clfcn.zero_grad()
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
                elif self.config['General']['model_specialization'] == 'cross':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_cross_scale(self.nclasses, output, annotation)
                else:
                    sys.exit("A specialization must be specified! (large or small or all or cross)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                if modality == 'rgb':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    train_loss += loss_rgb.item()
                    loss_rgb.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train rgb loss:{loss_rgb:.4f}')

                elif modality == 'lidar':
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    train_loss += loss_lidar.item()
                    loss_lidar.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train lidar loss:{loss_lidar:.4f}')

                elif modality == 'cross_fusion':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    loss_fusion = self.criterion(outputs['cross_fusion'], batch['anno'])
                    loss_all = loss_rgb + loss_lidar + loss_fusion
                    train_loss += loss_all.item()
                    loss_all.backward()
                    self.optimizer_clfcn.step()
                    progress_bar.set_description(f'train fusion loss:{loss_all:.4f}')

            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print( f'Training IoU for Epoch: {train_epoch_IoU}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i+1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clfcn(valid_dataloader, modality)

            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss,
                                        'valid': valid_epoch_loss}, epoch)

            writer.close()

            early_stop_index = round(valid_epoch_loss, 4)
            early_stopping(early_stop_index, epoch, self.model, modality, self.optimizer_clfcn)
            save_epoch = self.config['General']['save_epoch']
            if (epoch + 1) % save_epoch == 0 and epoch > 0:
                print(f'Saving model for every {save_epoch} epochs...')
                save_model_dict(self.config, epoch, self.model, modality, self.optimizer_clfcn, True)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
        print('Training Complete')

    def validate_clfcn(self, valid_dataloader, modal):
        """
        The validation of one epoch
        """
        self.model.eval()
        print('Validating...')
        modality = modal
        valid_loss = 0.0
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
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
                elif self.config['General']['model_specialization'] == 'cross':
                    batch_overlap, batch_pred, batch_label, batch_union = \
                        metrics.find_overlap_cross_scale(self.nclasses, output, annotation)
                else:
                    sys.exit("A specialization must be specified! (large or small or all or cross)")

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                if modality == 'rgb':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    valid_loss += loss_rgb.item()
                    progress_bar.set_description(f'valid rgb loss:{loss_rgb:.4f}')

                elif modality == 'lidar':
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    valid_loss += loss_lidar.item()
                    progress_bar.set_description(f'valid lidar loss:{loss_lidar:.4f}')

                elif modality == 'cross_fusion':
                    loss_rgb = self.criterion(outputs['rgb'], batch['anno'])
                    loss_lidar = self.criterion(outputs['lidar'], batch['anno'])
                    loss_fusion = self.criterion(outputs['cross_fusion'], batch['anno'])
                    loss_all = loss_rgb + loss_lidar + loss_fusion
                    valid_loss += loss_all.item()
                    progress_bar.set_description(f'valid fusion loss:{loss_all:.4f}')
        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        print(f'Validatoin IoU for Epoch: {valid_epoch_IoU}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i+1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        return valid_epoch_loss, valid_epoch_IoU
