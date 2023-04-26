# -*- encoding: utf-8 -*-
'''
@File    :   TF_rank.py
@Time    :   2021/12/30 19:19:43
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   put the wrong answer behind the right answer
'''


import json
from tqdm import tqdm
from itertools import chain


target = "/data1/zhifli/Models/proto-qa/benchmark/idea/gpt2-baseline.jsonl.judge" # xxx-count.jsonl

output = "/data1/zhifli/Models/proto-qa/benchmark/idea/gpt2-baseline.golden.jsonl"

f = open(target, 'r')
content = f.readlines()
examples = [json.loads(line) for line in content]

f.close()

# filter negative and positive case
data_set = []
for case in tqdm(examples):
    case_id = case['id']
    right = case['right']
    wrong = case['wrong']
    data_set.append({case_id:list(chain(right, wrong))})

f = open(output, 'w', encoding='utf-8')
for case in data_set:
    f.write(json.dumps(case) + '\n')

f.close()