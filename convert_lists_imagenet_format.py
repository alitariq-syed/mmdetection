#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:06:01 2021

@author: ali
"""

import numpy as np

base_dir = './data/TBX11K/lists/'

lists = ["TBX11K_val.txt","TBX11K_train.txt","TBX11K_trainval.txt","all_train.txt","all_trainval.txt","all_val.txt","all_test.txt"]
lists_out = ["TBX11K_val_imagenet.txt","TBX11K_train_imagenet.txt","TBX11K_trainval_imagenet.txt","all_train_imagenet.txt","all_trainval_imagenet.txt","all_val_imagenet.txt","all_test_imagenet.txt"]
TBX11K_val = np.loadtxt(base_dir+lists[0],dtype='U64')
TBX11K_train = np.loadtxt(base_dir+lists[1],dtype='U64')
TBX11K_trainval = np.loadtxt(base_dir+lists[2],dtype='U64')

all_train = np.loadtxt(base_dir+lists[3],dtype='U64')
all_trainval = np.loadtxt(base_dir+lists[4],dtype='U64')
all_val = np.loadtxt(base_dir+lists[5],dtype='U64')
all_test = np.loadtxt(base_dir+lists[6],dtype='U64')

dfs = [TBX11K_val,TBX11K_train, TBX11K_trainval, all_train, all_trainval, all_val, all_test]

for df in range(len(dfs)):
    for i in range(len(dfs[df])):
        if "shenzhen" in dfs[df][i]:
            if dfs[df][i][-5] == "1":
                dfs[df][i] = dfs[df][i]+" 2"
            else:
                dfs[df][i] = dfs[df][i]+" 0"
        elif "da+db" in dfs[df][i]:
            if dfs[df][i].split('/')[-1][0] == "p":
                dfs[df][i] = dfs[df][i]+" 2"
            else:
                dfs[df][i] = dfs[df][i]+" 0"
        else:
            if dfs[df][i][0]=="h":
                dfs[df][i] = dfs[df][i]+" 0"
            
            elif dfs[df][i][0]=="s":
                dfs[df][i] = dfs[df][i]+" 1"
            
            else:
                dfs[df][i] = dfs[df][i]+" 2"
    
    np.savetxt(fname=base_dir+lists_out[df],X=dfs[df], fmt='%s')