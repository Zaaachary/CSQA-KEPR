# -*- encoding: utf-8 -*-
'''
@File    :   split_dataset.py
@Time    :   2021/12/31 11:22:35
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   load unsplit data and split it to train test
'''
import random

target = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/v4.1/"

f = open(target+'data_unsplit.tsf', 'r')
unsplit = f.readlines()
f.close()

case_id_list = [case.split('\t')[0] for case in unsplit]
dev_id_list = random.sample(case_id_list, 100)
dev_id_list.sort(reverse=True)

dev_set = []
train_set = []
for case in unsplit:
    case_id = case.strip().split('\t')[0]
    if case_id in dev_id_list:
        dev_set.append(case)
    else:
        train_set.append(case)

f = open(target+'train.tsf', 'w')
for line in train_set:
    f.write(line)
f.close()

f = open(target+'dev.tsf', 'w')
for line in dev_set:
    f.write(line)
f.close()