# -*- encoding: utf-8 -*-
"""
@File    :   run_infer.py
@Time    :   2021/10/13 23:45:30
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   加载模型推断结果

"""
import os
import argparse
import pdb
import logging
import json
from collections import Counter
from re import L
import sys

from nltk.corpus import stopwords
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
import numpy as np
from data import CommonGen_Dataset, ProtoQA_Dataset

sys.path.append('../')
from Tools.data_io_util import dump_data
from model_wrapper import Generation_Model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  

en_stopwords = set(stopwords.words('english'))
en_stopwords.update(['their', 'her', 'his', 'ours'])

def prepare_inputs(args, raw_text, tokenizer, device):
    if args.model_type == 'gpt2':
        if "keyword_wkdt" in args.experiment:
            question, description = raw_text
            question_tokens = tokenizer.encode(description, question, max_length=args.max_src_len, truncation='only_first', add_special_tokens=False, return_tensors='pt')
        else:
            question_tokens = tokenizer.encode(raw_text[0], add_special_tokens=False, return_tensors='pt')
        question_tokens = question_tokens.to(device)
        question_len = len(question_tokens[0])
        faeture_dict = {'input_ids':question_tokens}
    elif args.model_type == 'bart':
        if "keyword_wkdt" in args.experiment:
            question, description = raw_text
            context_tokens = tokenizer.encode(description, question + tokenizer.mask_token, add_special_tokens=True, max_length=args.max_src_len, truncation='only_first')
        else:
            question = raw_text[0]
            context_tokens = tokenizer.encode(question + tokenizer.mask_token, add_special_tokens=True, max_length=args.max_src_len, truncation='only_first')
        
        input_ids = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
        decoder_input_ids = input_ids[:, :-2]
        question_len = len(input_ids[0]) -2
            
        faeture_dict = {'input_ids':input_ids, 'decoder_input_ids':decoder_input_ids}

    return faeture_dict, question_len

def prepare_commongen_inputs(args, raw_text, tokenizer, device):
    if 'ke' in args.experiment or 'proto' in args.experiment:
        description, concept = raw_text
        context_tokens = tokenizer.encode(description, concept, add_special_tokens=True, max_length=args.max_src_len, truncation='longest_first')
    else:
        concept_set = raw_text[0]
        context_tokens = tokenizer.encode(concept_set, add_special_tokens=True, max_length=args.max_src_len, truncation=True)
    
    input_ids = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
    faeture_dict = {'input_ids':input_ids}

    return faeture_dict


def run_beam_search(args, dataset, tokenizer, beam_nums, sample_nums, model, device):
    max_length = args.eval_length
    all_result_list = []
    all_info_list = []

    if args.commongen:
        for index, example in tqdm(enumerate(dataset), total=len(dataset)):
            example_ids, raw_text = example[0], example[1]
            if 'ke' in  args.experiment:
                concept_set = raw_text[1]
            else:
                concept_set = raw_text[0]
            result = {"concept_set": concept_set, "pred_scene": []}
            info = {'example_id': example_ids, 'inputs': raw_text, 'sentence':{}}
            
            feature_dict = prepare_commongen_inputs(args, raw_text, tokenizer, device)
            
            outputs = model.generate(
                num_beams=beam_nums, num_return_sequences=sample_nums, 
                max_length=max_length,
                output_scores = True,
                return_dict_in_generate= True,
                **feature_dict
            )
            outputs_token = outputs['sequences']
            score_list = outputs['sequences_scores']
            sentence_set = set()
            for sentence, score in zip(outputs_token, score_list):
                sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                if sentence in sentence_set:
                    continue
                else:
                    sentence_set.add(sentence)
                    result["pred_scene"].append(sentence)
                    info['sentence'][sentence] = round(score.item(), 5)
            all_result_list.append(result)
            all_info_list.append(info)
            # import pdb; pdb.set_trace()
            
        return all_result_list, all_info_list
    else:
        for index, example in tqdm(enumerate(dataset), total=len(dataset)):
            # prepare input
            example_ids, raw_text = example[0], example[1:]
            result = {example_ids: []}
            info = {"example_id": example_ids, 'inputs': raw_text, 'answer':{}}
            feature_dict, question_len = prepare_inputs(args, raw_text, tokenizer, device)
            # run model
            outputs = model.generate(
                num_beams=beam_nums, num_return_sequences=sample_nums, 
                max_length=max_length,
                output_scores = True,
                return_dict_in_generate= True,
                **feature_dict
            )
            outputs_token = outputs['sequences']
            answer_token_list = outputs_token[:,question_len:]
            answer_score_list = outputs['sequences_scores']
            # clean answer and info
            answer_set = set()


            for answer, score in zip(answer_token_list, answer_score_list):
                answer = tokenizer.decode(answer, skip_special_tokens=True)
                answer = answer.replace('.', '')
                answer_tokens = answer.strip().split(' ')[:4]       # avoid too long
                answer_tokens = [token for token in answer_tokens if token not in en_stopwords]
                answer = ' '.join(answer_tokens)
                if answer in answer_set:
                    continue
                else:
                    answer_set.add(answer)
                    result[example_ids].append(answer)
                    info['answer'][answer] = round(score.item(), 5)
            all_result_list.append(result)
            all_info_list.append(info)

        return all_result_list, all_info_list

def load_model(model_path, args):
    logging.info(f'load model from <{model_path}>')
    model = Generation_Model.load_from_checkpoint(
        PTM_name_or_path=args.PTM_name_or_path,
        checkpoint_path=model_path
        )
    tokenizer = model.get_tokenizer()
    # checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, tokenizer

def merge_answer(all_info_list):
    info_dict = {}
    for info in all_info_list:
        example_ids = info['example_ids']
        if example_ids in info_dict:
            info_dict[example_ids]['inputs'].append(info['inputs'][-1])
            info_dict[example_ids]['answer2'] = info['answer']
        else:
            info_dict[example_ids] = info
    
    result_list, info_list = [], []
    for example_ids, info in info_dict.items():
        if 'answer2' not in info:
            info['merged_answer'] = info['answer']
            continue
        info['merged_answer'] = {}
        for answer, score in info['answer'].items():
            if answer in info['answer2']:
                score = (score + info['answer2'][answer]) / 2
            info['merged_answer'][answer] = score
        for answer, score in info['answer2'].items():
            if answer not in info['merged_answer']:
                info['merged_answer'][answer] = score
        info_list.append(info)
        answer_list = list(info['merged_answer'].items())
        answer_list.sort(key=lambda x:x[1], reverse=True)
        result_list.append({example_ids: [ans[0] for ans in answer_list]})

    return result_list, info_list

def set_model_and_output_path(args):
    # infer all ckpt if model_path is ckpt dir
    top_path = args.model_path
    if os.path.isfile(top_path):
        ckpt_path = [top_path, ]
    elif os.path.isdir(top_path):
        ckpt_name_list = os.listdir(top_path)
        ckpt_name_list = [name for name in ckpt_name_list if name[-5:] == '.ckpt']
        if args.last_model:
            max_step, max_ckpt = 0, 0
            for index, ckpt in enumerate(ckpt_name_list):
                step = int(ckpt.split('-')[1].replace('step=',''))
                if step > max_step:
                    max_ckpt = index
                    max_step = step
            ckpt_name_list = [ckpt_name_list[max_ckpt]]
        ckpt_path = [os.path.join(top_path, ckpt_name) for ckpt_name in ckpt_name_list]
    else:
        logging.error('args.model_path not exists')
        return 

    # define output path
    if args.output_path:
        top_output_path = args.output_path
        if os.path.isfile(top_output_path):
            logging.error('args.output_path should be dir')
            return
    else:
        if os.path.isfile(args.model_path):
            logging.error('args.output_path required if args.model_path is not dir')
            return 
        else:
            top_output_path = os.path.join(args.model_path, '../generate_result')
            os.mkdir(top_output_path)
    
    output_path = []
    for path in ckpt_path:
        output_name = os.path.basename(path)[:-5] + '.jsonl'
        output_path.append(os.path.join(top_output_path, output_name))

    return ckpt_path, output_path

def main(args):
    if args.device >= 0:
        torch.cuda.set_device(args.device)
        device = 'cuda'
    else:
        device = 'cpu'
    ckpt_path_list, output_path_list = set_model_and_output_path(args)

    for ckpt_path, output_path in zip(ckpt_path_list, output_path_list):
        # load model
        model, tokenizer = load_model(ckpt_path, args)
        model.to(device)

        # dev/test Dataset
        logging.info(f'load dataset from <{args.dataset_path}>')
        if args.commongen:
            dataset = CommonGen_Dataset(
                args.dataset_path, None, None, tokenizer, 'predict', evaluate=True,
                experiment = args.experiment
            )
        else:
            dataset = ProtoQA_Dataset(
                args.dataset_path, args.model_type, None, None, 
                True, tokenizer, "predict", evaluate=True,
                experiment=args.experiment, wkdt_path=args.wkdt_path
            )

        # run generation
        logging.info(f'run generation')

        all_result_list, all_info_lsit = run_beam_search(args, dataset, tokenizer, args.beam_nums, args.sample_nums, model, device)
        if 'different_cs' in args.experiment:
            all_result_list, all_info_lsit = merge_answer(all_info_lsit)

        logging.info(f'save result to < {output_path} >')
        dump_data(all_result_list, output_path, mode='jsonl')
        dump_data(all_info_lsit, output_path[:-5]+'info.json', mode='json')
        # dump_data(all_info_lsit, output_path[:-5]+'info.jsonl', mode='json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_method", choices=['beam_search'], default='beam_search')
    parser.add_argument("--beam_nums", type=int, default=1)
    parser.add_argument("--sample_nums", type=int, default=1)
    parser.add_argument("--eval_length", type=int, default=60)
    parser.add_argument("--max_src_len", type=int, default=None,
        help='Bart source_length, GPT2 seq_len')
    
    parser.add_argument("--experiment", type=str, default='')
    parser.add_argument("--model_type", type=str, default='gpt2')
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--commongen", action='store_true', default=False)

    parser.add_argument("--last_model", action='store_true', default=False)
    parser.add_argument("--model_path", type=str, required=True, help='model ckpt or ckpt dir')
    parser.add_argument("--output_path", type=str, default=None, help='empty or a dir')
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--wkdt_path", type=str, default='')
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    
    args_str = """
        --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/dev.crowdsourced.jsonl\
        --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/param-7-split-answer/version_0/checkpoints \
        --PTM_name_or_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/gpt2-large \
        --experiment \ 
        --device -1 \
        --beam_nums 50 \
        --ranking_method None \
        --sample_nums 30
    """
        # --mc_freq
        # --dataset_path /home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/statistic

    args = parser.parse_args()
    # args = parser.parse_args(args_str.split())
    logging.info(args)
    main(args)