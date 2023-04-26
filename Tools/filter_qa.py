# -*- encoding: utf-8 -*-
'''
@File    :   filter_qa.py
@Time    :   2022/08/29 14:45:27
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
import os
from data_io_util import load_data, dump_data


def clean_data(data):
    result = []
    for item in data:
        question = item['question']['normalized']
        answers = list(item['answers']['raw'].keys())
        result.append([question, ])
        result.append([' || '.join(answers),])
    return result

def make_data():
    target = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/training"
    output = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/output/merge{}.tsf"
    
    
    data_list = []
    for file_name in os.listdir(target):
        if not file_name.endswith('.jsonl'):
            continue
        path = os.path.join(target, file_name)
        data = load_data(path, 'jsonl')
        data = clean_data(data)
        data_list.extend(data)
    
    temp_list = []
    MOD = 5000
    idx = 0
    file_idx = 0
    for case in data_list:
        # import pdb; pdb.set_trace()
        if idx < MOD:
            idx += 1
            temp_list.append(case)
        else:
            dump_data(temp_list, output.format(file_idx), 'tsf')
            file_idx += 1
            idx = 0
            temp_list.clear()
            temp_list.append(case)
    else:
        if temp_list:
            dump_data(temp_list, output.format(file_idx), 'tsf')

def split_data():
    target = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/training/merge.txt"
    data = load_data(target, 'plain')
    
    temp = dict()
    flag = True
    merge_data = []
    idx = 0
    temp['id'] = idx
    for example in data:
        if flag:
            temp['question'] = example.strip()
            flag = False
        else:
            temp['answers'] = [item.strip() for item in example.strip().split('||')]
            merge_data.append(temp)
            idx += 1
            temp = dict()
            temp['id'] = idx
            flag = True
    dump_data(merge_data, target+'1', 'jsonl')
            
        

if __name__ == "__main__":
    # target = ["/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/training", "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate"]
    
    # make_data()
    split_data()

            