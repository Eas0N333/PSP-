#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""
import os
import random
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import FewShot as FewShotSeg
from dataloader.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex

import matplotlib.pyplot as plt

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # 断点续训:加载模型/优化器状态,并把 lr 恢复到对应里程碑后再开始
    if _config['reload_model_path'] is not None:
        _log.info(f'Resuming from {_config["reload_model_path"]} (global iter {_config["start_iter"]})')
        _ckpt = torch.load(_config['reload_model_path'], map_location='cpu')
        if isinstance(_ckpt, dict) and 'model' in _ckpt:
            model.load_state_dict(_ckpt['model'])
            if 'optimizer' in _ckpt:
                optimizer.load_state_dict(_ckpt['optimizer'])
        else:
            model.load_state_dict(_ckpt)
        _lr_done = _config['start_iter'] // _config['max_iters_per_load']
        for _g in optimizer.param_groups:
            _g['lr'] = _config['optim']['lr'] * (_config['lr_step_gamma'] ** _lr_done)
        _log.info(f'Resumed, lr = {optimizer.param_groups[0]["lr"]:.6f}')

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'coarse_loss': 0, 'shape_loss': 0, 'rec_loss': 0, 'edge_loss': 0}

    i_iter = _config['start_iter']
    _stop = _config['start_iter'] + _config['iter_budget'] if _config['iter_budget'] > 0 else None
    _log.info(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        if _stop is not None and i_iter >= _stop:
            break
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):
            if _stop is not None and i_iter >= _stop:
                break
            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            # Compute outputs and losses.
            qry_pred_logit, query_loss,coarse_loss, shape_loss, rec_loss = model(support_images[0][0], query_images[0], support_fg_mask[0][0],query_labels, mode='train')
            loss = query_loss + coarse_loss + shape_loss + rec_loss
            # show_img([torch.argmax(qry_pred_logit, dim=1), query_labels], row=1)
            # show_img(show_list)
            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log lossw
            query_loss = query_loss.detach().data.cpu().numpy()
            shape_loss = shape_loss.detach().data.cpu().numpy()
            coarse_loss = coarse_loss.detach().data.cpu().numpy()
            rec_loss = rec_loss.detach().data.cpu().numpy()
            # edge_loss = edge_loss.detach().data.cpu().numpy()


            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss.item()
            log_loss['coarse_loss'] += coarse_loss.item()
            log_loss['shape_loss'] += shape_loss
            log_loss['rec_loss'] += rec_loss

            # log_loss['rec_loss'] += rec_loss
            # log_loss['edge_loss'] += edge_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                shape_loss = log_loss['shape_loss'] / _config['print_interval']
                coarse_loss = log_loss['coarse_loss'] / _config['print_interval']
                rec_loss = log_loss['rec_loss'] / _config['print_interval']
                # edge_loss = log_loss['edge_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['coarse_loss'] = 0
                log_loss['shape_loss'] = 0
                log_loss['rec_loss'] = 0
                log_loss['edge_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, coarse_loss:{coarse_loss}, shape_loss:{shape_loss} '
                          f'rec_loss:{rec_loss}')

                # _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, shape_loss: {shape_loss}'
                #           f', rec_loss: {rec_loss}, edge_loss: {edge_loss}, proto_loss: {proto_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
                # 额外存一份可续训的状态(模型+优化器+全局步数)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter': i_iter + 1},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'resume_{i_iter + 1}.pth'))

            i_iter += 1
    # 分块训练收尾:无论是否落在 save_snapshot_every 的整数倍上,都存一份可续训状态
    if i_iter % _config['save_snapshot_every'] != 0:
        torch.save(model.state_dict(),
                   os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter}.pth'))
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': i_iter},
                   os.path.join(f'{_run.observers[0].dir}/snapshots', f'resume_{i_iter}.pth'))
        _log.info(f'Saved tail snapshot at iter {i_iter}')
    _log.info('End of training.')
    return 1
