import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
import scipy.misc
from model import PointFlow
from torch import optim
from torch.backends import cudnn
import datahandling
# from arguments import get_args
from utils import AverageValueMeter, visualize_point_clouds, get_args
from tensorboardX import SummaryWriter

faulthandler.enable()


def main_worker():
    args = get_args()
    cudnn.benchmark = True
    model = PointFlow()

    start_epoch = 0    

    tr_dataset, te_dataset = datahandling.get_datasets()
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=16, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)
    scheduler = optim.lr_scheduler.ExponentialLR(model.optimizer)
    ### training
    start_time = time.time()
    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()
    for epoch in range(start_epoch, 15):
        print(epoch)
        # adjust the learning rate
        if (epoch + 1) % model.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)

        # train 1 epoch
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            step = bidx + len(train_loader) * epoch
            model.train()
            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            out = model(inputs, model.optimizer, step)
            entropy, prior_nats, recon_nats = out['entropy'], out['prior_nats'], out['recon_nats']
            entropy_avg_meter.update(entropy)
            point_nats_avg_meter.update(recon_nats)
            latent_nats_avg_meter.update(prior_nats)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()

        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            num_samples = min(10, inputs.size(0))
            num_points = inputs.size(1)
            _, samples = model.sample(num_samples, num_points)
            results = []
            for idx in range(num_samples):
                res = visualize_point_clouds(samples[idx], inputs[idx], idx,
                                             pert_order=train_loader.dataset.display_axis_order)
                results.append(res)
            res = np.concatenate(results, axis=1)
                


def main():
    # command line args
    main_worker()


if __name__ == '__main__':
    main()