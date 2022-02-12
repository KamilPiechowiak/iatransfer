from typing import Dict
import os
import shutil
import math
import numpy as np
import logging

import torch
from torch import nn, optim

from iatransfer.utils.file_utils import save_json, read_json
from iatransfer.researchput.models import get_model
from iatransfer.researchput.utils import get_path, get_teacher_model_path
from iatransfer.researchput.data import get_dataset
from iatransfer.researchput.metrics.stats_reporter import StatsReporter
from iatransfer.toolkit import IAT

def single_epoch(device, model, loader, loss_func, optimizer=None, stats=None, metrics={}, gradeBy='loss', gradient_accumulation=1):
    assert gradeBy in metrics.keys()
    is_training = (optimizer is not None)
    metric_values = {}
    for metric in metrics.keys():
        metric_values[metric] = []
    samples = []

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        logging.info(f"Loss: {loss.cpu().detach()}")
        for metric, f in metrics.items():
            metric_values[metric].append(f(y_pred, y).cpu().detach())
        samples.append(x.shape[0])
        if is_training:
            loss.backward()
            if i%gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

    if is_training and i%gradient_accumulation != 0:
        optimizer.step()
        optimizer.zero_grad()

    samples = np.array(samples)
    metric_keys = list(metric_values.keys())
    for key in metric_keys:
        metric_values[key] = np.sum(np.array(metric_values[key])*samples)/np.sum(samples)
    stats.update(metric_values, is_training=is_training)

    return metric_values[gradeBy]

def save_model(model, optimizer, scheduler, paths):
    for path in paths:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, path)

def train(config: Dict) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(device)

    model = get_model(config["model"], config["init"])
    
    if config["method"] is not None:
        teacher_model_path = get_teacher_model_path(config)
        teacher_config = read_json(os.path.join(teacher_model_path, "config.json"))
        teacher_model = get_model(teacher_config["model"], teacher_config["init"])
        checkpoint_filename = config.get("checkpoint", "best.pt")
        config["checkpoint"] = checkpoint_filename
        teacher_model.load_state_dict(
            torch.load(f"{teacher_model_path}/{checkpoint_filename}")['model']
        )
        iat = IAT(**config["method"])
        iat(teacher_model, model)
    
    model_path = get_path(config)
    save_json(config, os.path.join(model_path, "config.json"))

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lf = lambda x: (1 + math.cos(x * math.pi / config['scheduler_mocked_epochs'])) / 2 * 0.9 + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_func = nn.CrossEntropyLoss()
    metrics = {
        'loss': loss_func,
        'acc': lambda input, target: (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
    }
    stats_reporter = StatsReporter(metrics, model_path)

    train_dataset, val_dataset = get_dataset(
        config["dataset"]["name"],
        config["dataset"]["resolution"],
        config["datasets_path"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = config["num_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config["batch_size"],
        shuffle = False,
        num_workers = config["num_workers"],
    )

    bestLoss = 1e10
    for epoch in range(config["epochs"]):
        logging.info(f"Epoch {epoch}")
        model.train()
        single_epoch(device,
            model,
            train_loader,
            loss_func,
            optimizer,
            stats_reporter,
            metrics,
            gradient_accumulation=config.get("gradient_accumulation", 1))

        with torch.no_grad():
            model.eval()
            loss = single_epoch(device,
                model,
                val_loader,
                loss_func,
                None,
                stats_reporter,
                metrics,
                gradient_accumulation=config.get("gradient_accumulation", 1))

            if loss < bestLoss:
                bestLoss = loss
                save_model(model, optimizer, scheduler, [f'{model_path}/best.pt'])

        save_model(model, optimizer, scheduler, [f'{model_path}/current.pt'])
        if epoch % config['persist_state_every'] == config['persist_state_every'] - 1 and os.path.exists(f'{model_path}/best.pt'):
            shutil.copy(f'{model_path}/best.pt', f'{model_path}/{epoch}_checkpoint.pt')

        scheduler.step()
