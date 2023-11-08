import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched, get_lr_sched_decay_rate
from optim.misc import build_optimizer

from config.default import get_config
from dataloaders.loader import build_dataloader

from polarnet.dataloaders.pcd_keystep_dataset import (
    PCDKeystepDataset, pcd_stepwise_collate_fn,
    ProcessedPCDKeystepDataset
)
from polarnet.models.pcd_unet import PointCloudUNet


import warnings
warnings.filterwarnings("ignore")

from polarnet.utils.slurm_requeue import init_signal_handler


dataset_factory = {
    'pre_pcd_keystep_stepwise': (ProcessedPCDKeystepDataset, pcd_stepwise_collate_fn),
    'pcd_keystep_stepwise': (PCDKeystepDataset, pcd_stepwise_collate_fn),
}



def main(config):
    config.defrost()
    default_gpu, n_gpu, device = set_cuda(config)
    # config.freeze()

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )

    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    if type(config.DATASET.taskvars) is str:
        config.DATASET.taskvars = [config.DATASET.taskvars]

    # load data training set
    dataset_class, dataset_collate_fn = dataset_factory[config.DATASET.dataset_class]

    dataset = dataset_class(**config.DATASET)
    data_loader, pre_epoch = build_dataloader(
        dataset, dataset_collate_fn, True, config
    )
    LOGGER.info(f'#num_steps_per_epoch: {len(data_loader)}')
    if config.num_train_steps is None:
        config.num_train_steps = len(data_loader) * config.num_epochs
    else:
        assert config.num_epochs is None, 'cannot set num_train_steps and num_epochs at the same time.'
        config.num_epochs = int(
            np.ceil(config.num_train_steps / len(data_loader)))

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # Prepare model
    model = PointCloudUNet(**config.MODEL)
    # DDP: SyncBN
    if int(os.environ['WORLD_SIZE']) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" %
                (model.num_trainable_parameters))
    
    config.freeze()

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.resume_training:
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']
        restart_epoch = global_step // len(data_loader)
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = config.restart_epoch
        global_step = restart_epoch * len(data_loader) 

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint, strict=config.checkpoint_strict_load)

    model.train()
    # set_dropout(model, config.dropout)
    model = wrap_model(model, device, config.local_rank)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.train_batch_size if config.local_rank == -
                1 else config.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.num_train_steps)

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    init_signal_handler()

    for epoch_id in range(restart_epoch, config.num_epochs):
        if global_step >= config.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        
        for step, batch in enumerate(data_loader):
            # forward pass
            losses, logits = model(batch, compute_loss=True)

            # backward pass
            if config.gradient_accumulation_steps > 1:  # average loss
                losses['total'] = losses['total'] / \
                    config.gradient_accumulation_steps
            losses['total'].backward()

            acc = ((logits[..., -1].data.cpu() > 0)
                   == batch['actions'][..., -1].cpu()).float()
                   
            if 'step_masks' in batch:
                acc = torch.sum(acc * batch['step_masks']) / \
                    torch.sum(batch['step_masks']).cpu()
            else:
                acc = acc.mean().cpu()

            for key, value in losses.items():
                TB_LOGGER.add_scalar(
                    f'step/loss_{key}', value.item(), global_step)
            TB_LOGGER.add_scalar('step/acc_open', acc.item(), global_step)

            # optimizer update and logging
            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                if config.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm
                    )
                    # print(step, name, grad_norm)
                    # for k, v in model.named_parameters():
                    #     if v.grad is not None:
                    #         v = torch.norm(v).data.item()
                    #         print(k, v)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (
                    lk, lv.item()) for lk, lv in losses.items()] + ['acc:%.2f' % (acc*100)]))
                LOGGER.info('===============================================')

            if global_step % config.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

            if global_step >= config.num_train_steps:
                break

    if global_step % config.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.item())
                    for lk, lv in losses.items()] + ['acc:%.2f' % (acc*100)]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument('--sleep_time', type=float, default=0, help='hour')
    parser.add_argument('--restart_epoch', type=int, default=0)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    if args.sleep_time > 0:
        time.sleep(args.sleep_time * 3600)

    config = get_config(args.exp_config, args.opts)

    config.defrost()
    config.restart_epoch = args.restart_epoch
    config.freeze()

    for i in range(len(config.CMD_TRAILING_OPTS)):
        if config.CMD_TRAILING_OPTS[i] == "DATASET.taskvars":
            if type(config.CMD_TRAILING_OPTS[i + 1]) is str:
                config.CMD_TRAILING_OPTS[i +
                                         1] = [config.CMD_TRAILING_OPTS[i + 1]]

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config


if __name__ == '__main__':
    config = build_args()
    main(config)
