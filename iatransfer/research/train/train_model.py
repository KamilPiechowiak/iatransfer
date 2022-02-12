from asyncio.log import logger
import math
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, List

import torch
from torch import nn, optim
from iatransfer.research.distributed.device_connector import DeviceConnector

from iatransfer.research.metrics.stats_reporter import StatsReporter
from iatransfer.research.train.model_freezer import ModelFreezer


def single_epoch(epoch: int, device: torch.device, connector: DeviceConnector,
                 model: nn.Module, loader: torch.utils.data.DataLoader, loss_func: Callable,
                 opt: torch.optim.Optimizer = None, stats: StatsReporter = None,
                 metrics: Dict = {}, gradeBy: str = 'bce', grad_acc: int = 1):
    start_time = datetime.now()
    assert gradeBy in metrics.keys()
    is_training = (opt is not None)
    metric_values = {}
    for metric in metrics.keys():
        metric_values[metric] = []

    for i, (x, y) in enumerate(loader):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        for metric, f in metrics.items():
            metric_values[metric].append(f(y_pred, y).detach())
            # if metric == "acc" and metric_values[metric][-1] < 0.01:  # TODO remove begin
            #     y_cpu = y_pred.cpu().detach()
            #     arr = [loss.cpu().detach(), y_cpu, y_cpu.min(), y_cpu.max(), y_cpu.mean()]
            #     connector.print(arr, flush=True)  # TODO remove end
        if is_training:
            loss.backward()
            # if epoch == 5:  # TODO remove begin
            #     connector.print(loss.detach(), flush=True)
            #     w_min, w_max = 0, 0
            #     for p in model.parameters():
            #         w_min = min(p.min().item(), w_min)
            #         w_max = max(p.max().item(), w_max)
            #     print([w_min, w_max, y_pred.min().item(), y_pred.max().item()], flush=True)  # TODO remove end
            if i % grad_acc == 0:
                connector.optimizer_step(opt)
                opt.zero_grad()
    if is_training and i % grad_acc != 0:
        connector.optimizer_step(opt)
        opt.zero_grad()

    metric_keys = list(metric_values.keys())
    metric_list = []
    for metric in metric_keys:
        metric_list.append(torch.tensor(metric_values[metric], device=device).mean())

    connector.all_avg(metric_list)
    for i, metric in enumerate(metric_keys):
        metric_values[metric] = metric_list[i].item()
    if connector.is_master() and stats is not None:
        stats.update(metric_values, is_training=is_training)

    end_time = datetime.now()
    connector.print(end_time - start_time, flush=True)
    return metric_values[gradeBy]


def save_model(connector: DeviceConnector, model: nn.Module,
               optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
               paths: List[str]):
    for path in paths:
        connector.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, path)


def train_model(FLAGS: Dict, device: torch.device, connector: DeviceConnector,
                model: nn.Module, path: str, train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset):
    os.makedirs(path, exist_ok=True)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=FLAGS['lr'])
    lf = lambda x: (1 + math.cos(x * math.pi / FLAGS['scheduler_mocked_epochs'])) / 2 * 0.9 + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    loss_func = nn.CrossEntropyLoss()
    metrics = {
        'loss': loss_func,
        'acc': lambda input, target: (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
    }
    if connector.is_master():
        statsReporter = StatsReporter(metrics, path)
    else:
        statsReporter = None

    train_sampler, val_sampler = connector.get_samplers(train_dataset, val_dataset)

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
        num_workers=FLAGS['num_workers'],
        drop_last=True)

    model_freezer = ModelFreezer()
    if FLAGS.get('freeze') is not None:
        model_freezer.freeze(model, FLAGS["model"]["args"][0])

    bestLoss = 1e10
    for epoch in range(FLAGS['epochs']):

        if FLAGS.get('unfreeze_at') == epoch:
            model_freezer.unfreeze(model)

        if hasattr(train_sampler, "set_epoch"):
            logger.info(f"Setting epoch {epoch}")
            train_sampler.set_epoch(epoch)
        connector.print(f'EPOCH: {epoch}')
        model.train()
        single_epoch(epoch, device, connector, model, connector.wrap_data_loader(train_loader, device), loss_func, opt,
                     stats=statsReporter, metrics=metrics, gradeBy='loss', grad_acc=FLAGS.get("grad_acc", 1))

        with torch.no_grad():
            model.eval()
            loss = single_epoch(epoch, device, connector, model, connector.wrap_data_loader(val_loader, device),
                                loss_func, stats=statsReporter, metrics=metrics, gradeBy='loss', grad_acc=FLAGS.get("grad_acc", 1))
            if loss < bestLoss:
                bestLoss = loss
                save_model(connector, model, opt, scheduler, [f'{path}/best.pt'])

        save_model(connector, model, opt, scheduler, [f'{path}/current.pt'])
        if epoch % FLAGS['persist_state_every'] == FLAGS['persist_state_every'] - 1 and connector.is_master() and os.path.exists(f'{path}/best.pt'):
            shutil.copy(f'{path}/best.pt', f'{path}/{epoch}_checkpoint.pt')

        scheduler.step()
    if connector.is_master():
        os.system(f'gsutil cp -r {path} {FLAGS["bucket_path"]}')
