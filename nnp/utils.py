#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:45:30 2019

@author: gianni
"""
import glob
import csv
import json
import os
import time
import argparse
import torch
import yaml
import sys
import numpy as np

def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LogWriter(object):
    #kind of inspired form openai.baselines.bench.monitor
    #We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header=''):
        self.keys = tuple(keys)+('t',)
        assert path is not None
        self._clean_log_dir(path)
        filename = os.path.join(path, 'monitor.csv')

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo['t'] = t
            self.logger.writerow(epinfo)
            self.f.flush()

    def _clean_log_dir(self,log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            files = glob.glob(os.path.join(log_dir, '*.csv'))
            for f in files:
                os.remove(f)


class LoadFromFile(argparse.Action):
    #parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__ (self, parser, namespace, values, option_string = None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            raise ValueError("configuration file must end with yaml or yml")
    


def save_argparse(args,filename,exclude=None):
    if filename.endswith('yaml') or filename.endswith('yml'):
        if isinstance(exclude, str):
            exclude = [exclude,]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, 'w'))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def group_weight(module, weight_decay):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
            try:
                group_decay.append(m.weight)
            except:
                pass
            try:
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            except:
                pass

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=weight_decay)]
    return groups


from sklearn.model_selection import train_test_split

def train_val_test_split(dset_len,val_ratio,test_ratio, seed, order=None):
    shuffle = True if order is None else False
    valtest_ratio = val_ratio+test_ratio
    idx_train = list(range(dset_len))
    idx_test = []
    idx_val = []
    if valtest_ratio>0 and dset_len>0:
        idx_train, idx_tmp = train_test_split(range(dset_len), test_size=valtest_ratio, random_state=seed, shuffle=shuffle)
        if test_ratio == 0:
            idx_val = idx_tmp
        elif val_ratio == 0:
            idx_test = idx_tmp
        else:
            test_val_ratio = test_ratio/(test_ratio+val_ratio)
            idx_val, idx_test = train_test_split(idx_tmp, test_size=test_val_ratio,random_state=seed, shuffle=shuffle)

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def set_batch_size(max_batch_size, len_dataset):
    batch_size = min(int(max_batch_size*len_dataset/32768)+4,max_batch_size)  #min size is equal to 4
    if batch_size != max_batch_size:
        print('Warning: Dataset lenght {}. Reducing batch_size {}'.format(len_dataset,batch_size))
    return batch_size

if __name__ == '__main__':
    order = [9,8,7,6,5,4,3,2,1,0]
    idx_train, idx_val, idx_test = train_val_test_split(10,0.1,0.2,0,order)
    print(idx_train)
    print(idx_val)
    print(idx_test)