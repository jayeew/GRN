#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import argparse
from dataset_utils import DataLoader
from utils import *
from models import *
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from other_models import *
from sklearn.metrics import roc_auc_score

def train(model, optimizer, data, args):
    model.train()
    optimizer.zero_grad()
    out, h = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    del out

def test(model, data, args):
    model.eval()
    accs, losses, preds = [], [], []
    out, h = model(data)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(dim=1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(out[mask], data.y[mask])
        # preds.append(pred.detach().cpu())
        accs.append(acc)
        # losses.append(loss.detach().cpu())
    return accs, h

def show_results(args, Results):
    test_acc_mean, val_acc_mean = np.mean(Results, axis=0) * 100 
    test_acc_std = np.sqrt(np.var(Results, axis=0)[0]) * 100 
    confidence_interval = 1.96 * test_acc_std/np.sqrt(10)
    print(f'On dataset {args.dataset}, in 10 repeated experiment:')
    print(f'Test acc mean= {test_acc_mean:.2f} ± {confidence_interval:.2f} \t val acc mean = {val_acc_mean:.2f}')

    # file = open(f'./save/onlyOutput_log.txt', 'a')
    # print(f'dataset : {args.dataset}, num_layers : {args.num_layers}:', file=file)
    # # print(f'num_layers:{args.num_layers},  dropout:{args.dropout}, lr:{args.lr}, weight_decay:{args.weight_decay}, hidden:{args.hidden}', file=file)
    # print(f'Test acc mean= {test_acc_mean:.2f} \t val acc mean = {val_acc_mean:.2f}', file=file)
    # print('*'*30, file=file)

def RunExp(args, dataset, data, Net, split):
    N = data.x.size(0)
    model = Net(dataset, args, N)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset in ["computers", "photo", "penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(args.val_rate*len(data.y)))
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb)
    else:
        data = geom_mask(args.dataset, data, split)
        
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_test_acc = 0, 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in trange(args.epochs):
        train(model, optimizer, data, args)
        [train_acc, val_acc, tmp_test_acc], h = test(model, data, args)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = tmp_test_acc
            best_epoch = iter
            # torch.save(h, f'save/{args.dataset}.pt')
            
    return best_test_acc, best_val_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=500)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--train_rate', type=float, default=0.5)
    parser.add_argument('--val_rate', type=float, default=0.25)
    parser.add_argument('--splits', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', default='penn94')
    args = parser.parse_args()

    set_seed(args.seed)
    Net = GRN
    
    dataset, data = DataLoader(args.dataset)
    
    Results = []
    for i in trange(args.splits):
        test_acc, best_val_acc = RunExp(args, dataset, data, Net, i)
        Results.append([test_acc, best_val_acc])
    show_results(args, Results)    
    
    
