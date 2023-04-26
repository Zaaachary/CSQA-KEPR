# -*- encoding: utf-8 -*-
'''
@File    :   run_infer.py
@Time    :   2022/01/05 16:11:00
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   
'''

import argparse
import os
import logging
from collections import Counter, OrderedDict
from torch.utils.data import dataloader

from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl 

from data import Ranking_Dataset
from model_wrapper import Ranking_Wrapper_Model
# from build_dataset.build_dataset import dump_data

import sys
sys.path.append('../')
from Tools.data_io_util import dump_data


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  

def gpus_parser(gpus):
    '''
    match input type with pytorch-lightning type
    "6,7" -> [6, 7]; "6" -> [6,];"-1" -> 0
    '''
    accelerator = None
    if gpus == "-1":
        gpus = 0            # not use gpu
    elif "," in gpus:
        # muliti gpu
        gpus = gpus.split(',')
        if '' in gpus:
            gpus.remove('')
        gpus = list(map(int, gpus))
        accelerator = 'ddp'
    else:
        # single gpu
        gpus = [int(gpus), ]
    return gpus, accelerator

def sort_and_save(info_dict, output_path, args):
    # sort answer
    result_dict = OrderedDict()  # save anser only (for evaluate)
    result_dict_2 = OrderedDict()  # save other information
    for example_id, answer_list in info_dict.items():
        answer_list.sort(key=lambda x:x[2], reverse=True)
        result_dict[example_id] = [ans[1] for ans in answer_list]
        result_dict_2[example_id] = answer_list

    # write for evaluate
    if 'commongen' not in args.experiment:
        result_list = []
        for key, value in result_dict.items():
            temp = {key:value}
            result_list.append(temp)
        dump_data(result_list, output_path, mode='jsonl')
    else:
        result_list = [None] * len(result_dict)
        for key, info in info_dict.items():
            concept_set = '#'.join(info[0][0].split(', '))
            temp = {'concept_set': concept_set, 'pred_scene': []}
            for inf in info:
                temp['pred_scene'].append(inf[1])
            result_list[key] = temp
            
    # write for read
    result_list = []
    for key, value in result_dict_2.items():
        example = {'example_id': key, 'inputs': [], 'answer':{}}
        question = value[0][0]
        example['inputs'].append(question)
            # example['inputs'].append(1)
        for item in value:
            if args.add_wkdt:
                _, answer, score, cs = item
                example['inputs'].append(cs)
            else:
                _, answer, score = item
            example['answer'][answer] = score
        result_list.append(example)
    dump_data(result_list, output_path[:-5]+'info.json', mode='json')

def set_model_and_output_path(args):
    # infer all ckpt if model_path is ckpt dir
    # import pdb; pdb.set_trace()
    if not args.multi_model:
        if args.last_model:
            assert os.path.isdir(args.model_path)
            max_step, max_ckpt = 0, 0
            ckpt_list = os.listdir(args.model_path)
            for index, ckpt in enumerate(ckpt_list):
                # import pdb; pdb.set_trace()
                step = int(ckpt.split('-')[1].replace('step=',''))
                if step > max_step:
                    max_ckpt = index
                    max_step = step
            ckpt_name = ckpt_list[max_ckpt]
            if not args.output_path:
                base_name = os.path.basename(args.target_path)
                args.output_path = os.path.join(args.model_path, '../', base_name)
            args.model_path = os.path.join(args.model_path, ckpt_name)
            assert os.path.isfile(args.model_path)
        
        if args.multi_target:
            # single model multi target
            output_path_list = []
            assert os.path.isdir(args.target_path)
            top_path = args.target_path
            target_name_list = os.listdir(top_path)
            target_name_list = [name for name in target_name_list if name[-6:] == '.jsonl']
            target_path_list = [os.path.join(top_path, target_name) for target_name in target_name_list]
            ckpt_path_list = [args.model_path] * len(target_path_list)
            
            top_output_path = args.output_path
            if not os.path.isdir(args.output_path):
                os.mkdir(top_output_path)

            for target_name in target_name_list:
                output_name = target_name[:-6] + '.jsonl'
                output_path_list.append(os.path.join(top_output_path, output_name))
        else:
            # single model single target
            assert not os.path.isdir(args.output_path)
            ckpt_path_list = [args.model_path, ]
            output_path_list = [args.output_path, ]
            target_path_list = [args.target_path, ]
    elif not args.multi_target:
        # multi model single target
        top_path = args.model_path
        assert os.path.isdir(args.model_path)
        ckpt_name_list = os.listdir(top_path)
        ckpt_name_list = [name for name in ckpt_name_list if name[-5:] == '.ckpt']
        ckpt_path_list = [os.path.join(top_path, ckpt_name) for ckpt_name in ckpt_name_list]

        # define output path
        output_path_list = []
        if args.output_path:
            assert os.path.isdir(args.output_path)
            top_output_path = args.output_path
        else:
            top_output_path = os.path.join(top_path, "../ranked_result")
            if not os.path.exists(top_output_path):
                os.mkdir(top_output_path)
            
        for ckpt_name in ckpt_name_list:
            output_name = ckpt_name[:-5] + '.jsonl'
            output_path_list.append(os.path.join(top_output_path, output_name))

        target_path_list = [args.target_path, ] * len(ckpt_path_list)
    return ckpt_path_list, output_path_list, target_path_list

def main(args):
    gpu, _ = gpus_parser(args.device)
    ckpt_path_list, output_path_list, target_path_list = set_model_and_output_path(args)

    # import pdb; pdb.set_trace()

    for ckpt_path, output_path, target_path in zip(ckpt_path_list, output_path_list, target_path_list):
        logging.info(f'load model from <{ckpt_path}>')
        # load model
        model = Ranking_Wrapper_Model.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            PTM_name_or_path=args.PTM_name_or_path,
            transfer_learning=None,
            )

        tokenizer = model.get_tokenizer()

        # load dataset
        dataset = Ranking_Dataset(
            dataset_path=args.dataset_path,
            max_seq_len = args.max_len,
            tokenizer = tokenizer,
            dataset_type="predict",
            overwrite_cache=True, 
            ranking_target=target_path,
            add_wkdt = args.add_wkdt,
            wkdt_path = args.wkdt_path,
            experiment=args.experiment,
            keyword_path=args.keyword_path
            )
        dataloader = dataset.make_dataloader(32)
        # model wrapper convert binary classication and regression output to a ranking score
        trainer = pl.Trainer(gpus=gpu, logger=False)
        result = trainer.predict(model, dataloader)
        if args.CLS_4CE:
            score = torch.stack(result[:-1])
            score = torch.cat((score, result[-1].unsqueeze(0)))
            score = score.reshape(-1,4)
        else:
            score = torch.flatten(torch.stack(result[:-1]))
            score = torch.cat((score, result[-1].reshape(-1)))
        score_list = score.tolist()

        # match score and ans
        info_dict = {}
        for example_id, example, score in zip(dataset.example_ids, dataset.raw_examples, score_list):
            info_dict[example_id] = info_dict.get(example_id, [])
            Q = example[0]
            # import pdb; pdb.set_trace()
            if args.add_wkdt:
                if 'keyword' in args.experiment:
                    cs = Q.split('[SEP]')[0]
                    Q = Q.replace(cs, '')
                    ans = example[1]
                    
                    if "? A:" in ans:
                        question, answer = ans.split("? A:")
                    else:
                        temp = ans.split(' is ')
                        answer = temp[-1]
                        question = ' is '.join(temp[:-1]) + ' is '
                    Q = Q + ' ' +question
                    ans = answer.replace('.', '')
                else:
                    ans_cs = example[1].split(' [SEP] ')
                    if len(ans_cs) == 2:
                        ans, cs = ans_cs
                    else:
                        ans, cs = ans_cs[0], 'None'
                    
                info_dict[example_id].append((Q, ans, score, cs))
            else:
                ans = example[1]
                info_dict[example_id].append((Q, ans, score))
        logging.info(f'save result to <{output_path}>')
        sort_and_save(info_dict, output_path, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--add_wkdt", action='store_true', default=False)
    parser.add_argument("--CLS_4CE", action='store_true', default=False)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--keyword_path", type=str, default='')
    parser.add_argument("--experiment", type=str, default='')

    parser.add_argument("--multi_model", action='store_true', default=False)
    parser.add_argument("--last_model", action='store_true', default=False)
    parser.add_argument("--multi_target", action='store_true', default=False)

    parser.add_argument("--model_path", type=str, required=True, help="trained ranking model")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, help='origin data path, provide question')
    parser.add_argument("--target_path", type=str, required=True, help='generated answer.jsonl')
    parser.add_argument("--wkdt_path", type=str, default='')
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    
    # --CLS_4CE
    # cu13
    # --add_wkdt
    args_str = """
    --max_len 120
    --multi_model

    --model_path /data1/zhifli/Models/proto-qa/ranking_model/electra_bce/data=v4_bz=1x8x1_ep=1_lr=5e-06_seed=42/checkpoints

    --target_path /home/zhifli/CODE/CS_Model_Adaptation/src/Finetune/outputs/gpt2_finetune/ranked_list.jsonl
    
    --dataset_path /home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/dev.crowdsourced.jsonl
    --wkdt_path /data1/zhifli/WKDT/wiktionary.pkl
    --PTM_name_or_path /data1/zhifli/PTMs/electra-large-discriminator
    --device 0
    """
    # szcs
        # --PTM_name_or_path /data1/zhifli/PTMs/electra-large-discriminator/
    # args_str = """
    #     --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/dev.crowdsourced.jsonl
    #     --target_path /data1/zhifli/Models/proto-qa/Gerneration_Model/without_pad_trans_ans/bz=4_ep=1_lr=5e-05_seed=42/generate_result/epoch=00-step=11750-val_loss=2.5823.jsonl
    #     --output_path /data1/zhifli/Models/proto-qa/Gerneration_Model/without_pad_trans_ans/bz=4_ep=1_lr=5e-05_seed=42/generate_result/epoch=00-step=11750-val_loss=2.5823.rank-3.jsonl
    #     --model_path /data1/zhifli/Models/proto-qa/ranking_model/data_v3/checkpoints/epoch=00-step=16193-val_loss=0.4345.ckpt
    #     --PTM_name_or_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/electra-large-discriminator
    #     --device 6
    # """

    args = parser.parse_args()
    # args = parser.parse_args(args_str.split())
    logging.info(args)
    main(args)