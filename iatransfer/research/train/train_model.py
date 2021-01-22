import math
import os
import shutil
from datetime import datetime
from time import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch import nn, optim
from torch.utils.data import Dataset

from iatransfer.research.metrics.stats_reporter import StatsReporter


def single_epoch(device, model, loader, loss_func, opt=None, stats=None, metrics={}, gradeBy='bce', grad_acc=1):
    start_time = datetime.now()
    assert gradeBy in metrics.keys()
    is_training = (opt is not None)
    metric_values = {}
    for metric in metrics.keys():
        metric_values[metric] = []

    i = 0
    t3 = time()
    a1, a2, a3 = 0, 0, 0
    for x, y in loader:
        i += 1
        t1 = time()
        a1 += t1 - t3
        y_pred = model(x)
        t2 = time()
        loss = loss_func(y_pred, y)
        for metric, f in metrics.items():
            metric_values[metric].append(f(y_pred, y).detach())
        if is_training:
            loss.backward()
            if i%grad_acc == 0:
                xm.optimizer_step(opt)
                opt.zero_grad()
        t3 = time()
        a2 += t2 - t1
        a3 += t3 - t2
    if is_training and i%grad_acc != 0:
        xm.optimizer_step(opt)
        opt.zero_grad()
    a1 /= i
    a2 /= i
    a3 /= i

    metric_keys = list(metric_values.keys())
    metric_list = []
    for metric in metric_keys:
        metric_list.append(torch.tensor(metric_values[metric], device=device).mean())

    xm.all_reduce(xm.REDUCE_SUM, metric_list)
    i = 0
    for metric in metric_keys:
        metric_values[metric] = (metric_list[i] / xm.xrt_world_size()).item()
        i += 1
    if xm.is_master_ordinal() and stats is not None:
        stats.update(metric_values, is_training=is_training)

    end_time = datetime.now()
    xm.master_print(end_time - start_time)
    return metric_values[gradeBy]


def save_model(model, optimizer, scheduler, paths):
    for path in paths:
        xm.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, path)


def train_model(FLAGS, device, model, path, train_dataset, val_dataset):
    os.makedirs(path, exist_ok=True)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=FLAGS['learning_rate'])
    lf = lambda x: (1 + math.cos(x * math.pi / FLAGS['scheduler_mocked_epochs'])) / 2 * 0.9 + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    loss_func = nn.CrossEntropyLoss()
    metrics = {
        'loss': loss_func,
        'acc': lambda input, target: (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
    }
    statsReporter = StatsReporter(metrics, path)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS['batch_size'],
        sampler=train_sampler,
        num_workers=FLAGS['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=FLAGS['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=FLAGS['num_workers'],
        drop_last=True)
    bestLoss = 1e10
    for epoch in range(FLAGS['epochs']):
        xm.master_print(f'EPOCH: {epoch}')
        model.train()
        single_epoch(device, model, pl.ParallelLoader(train_loader, [device]).per_device_loader(device), loss_func, opt,
                     stats=statsReporter, metrics=metrics, gradeBy='loss', grad_acc=FLAGS.get("grad_acc", 1))

        with torch.no_grad():
            model.eval()
            loss = single_epoch(device, model, pl.ParallelLoader(val_loader, [device]).per_device_loader(device),
                                loss_func, stats=statsReporter, metrics=metrics, gradeBy='loss', grad_acc=FLAGS.get("grad_acc", 1))
            if loss < bestLoss:
                bestLoss = loss
                save_model(model, opt, scheduler, [f'{path}/best.pt'])

        save_model(model, opt, scheduler, [f'{path}/current.pt'])
        if epoch % FLAGS['persist_state_every'] == FLAGS['persist_state_every'] - 1 and xm.is_master_ordinal() and os.path.exists(f'{path}/best.pt'):
            shutil.copy(f'{path}/best.pt', f'{path}/{epoch}_checkpoint.pt')

        scheduler.step()
    if xm.is_master_ordinal():
        os.system(f'gsutil cp -r {path} {FLAGS["bucket_path"]}')
