# -*- encoding: utf-8 -*-
'''
@File    :   build_dataset_v2.py
@Time    :   2022/03/13 19:21:02
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
from audioop import reverse
from operator import index
import os, sys
import pdb
import json
import math
import random
import copy
import argparse
from typing import List
from torch import threshold

from tqdm import tqdm
from itertools import chain
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from torch.nn import functional as F


sys.path.append('../')
from Tools.data_io_util import load_data, dump_data
# lemmatizer = WordNetLemmatizer()


def make_dataset(origin_data):
    target_count = 1
    
    result = []      # concept, sentence, score
    total_data_set = set()
    for data in origin_data:
        for sentence in data['scene']:
            total_data_set.add(sentence)
    
    total_data_set = list(total_data_set)
    for idx, data in enumerate(origin_data):
        count = 0
        concept = 'Make a sentence with "' + ', '.join(data['concept_set'].split('#')) + '": '
        for sentence in data['scene']:
            if count == target_count:
                break
            count += 1
                
            result.append(['id'+str(idx), concept, sentence, 1])
            negative = random.choice(total_data_set)
            while negative in data['scene']:
                negative = random.choice(total_data_set)
            result.append(['id'+str(idx), concept, negative, 0])
                
    return result

def main(args):
    output_top = os.path.join(args.output_path, args.dataset_name)
    if not os.path.exists(output_top):
        os.mkdir(output_top)
    output_top = os.path.join(output_top, args.dataset_version)
    if not os.path.exists(output_top):
        os.mkdir(output_top)


    for datatype in ['dev', 'train']:
        origin_path = os.path.join(args.origin_dataset, f"{datatype}.jsonl")
        output_path = os.path.join(output_top, f"{datatype}.tsf")
    
        origin_data = load_data(origin_path, 'jsonl')
        dataset = make_dataset(origin_data)
        dump_data(dataset, output_path, 'tsf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_version", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--test", action='store_true')

    args_str = r"""
    --origin_dataset /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/CommonGen/description_v2
    --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/CommonGen
    --dataset_name random
    --dataset_version v1
    """
    args = parser.parse_args(args_str.split())
    print(args)
    random.seed(42)
    main(args)