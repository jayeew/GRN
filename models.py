#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Linear
from torch_geometric.utils import degree, index_sort, to_dense_adj
from torch_sparse import SparseTensor
from utils import *

    
class GRN(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(GRN, self).__init__()
        self.dropout = args.dropout
        self.N = N
        self.w11=Linear(N, args.hidden)
        self.w22=Linear(dataset.num_features, args.hidden)

        self.w3=Linear(args.hidden, args.hidden)
        self.w4=Linear(args.hidden, args.hidden)

        self.out=Linear(args.hidden, dataset.num_classes)


    @classmethod
    def _norm(cls, edge_index):
        adj = to_dense_adj(edge_index).squeeze()
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        return adj
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        adj_ = SparseTensor(row=edge_index[0], col=edge_index[1],	
                        sparse_sizes=(x.size(0), x.size(0))
                                ).to_torch_sparse_coo_tensor()
        
        adj=self.w11(adj_)
        x=self.w22(x)
        h1=torch.mul(adj, x)
        h1=F.sigmoid(h1)
        h=self.out(h1)
        return F.log_softmax(h, dim=1), h
    
class Model1(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(Model1, self).__init__()
        self.dropout = args.dropout
        self.N = N
        self.w11=Linear(N, args.hidden)
        self.w22=Linear(dataset.num_features, args.hidden)
        self.out=Linear(args.hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x=self.w22(x)
        h=x
        h=self.out(h)
        return F.log_softmax(h, dim=1), h
    
class Model2(torch.nn.Module):
    def __init__(self, dataset, args, N):
        super(Model2, self).__init__()
        self.dropout = args.dropout
        self.N = N
        self.w11=Linear(N, args.hidden)
        self.w22=Linear(dataset.num_features, args.hidden)
        self.out=Linear(args.hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj_ = SparseTensor(row=edge_index[0], col=edge_index[1],	
                        sparse_sizes=(x.size(0), x.size(0))
                                ).to_torch_sparse_coo_tensor()
        
        adj=self.w11(adj_)
        h=self.out(adj)
        return F.log_softmax(h, dim=1), h