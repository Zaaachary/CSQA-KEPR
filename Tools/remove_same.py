# -*- encoding: utf-8 -*-
'''
@File    :   judge_result.py
@Time    :   2022/03/22 14:07:01
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
import json
from functools import partial
from textwrap import indent
from collections import OrderedDict
from data_io_util import load_data, dump_data
from typing import *
from tqdm import tqdm

import numpy as np
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import general_eval, evaluate
from protoqa_evaluator.scoring import wordnet_score

max_answers = {
    f"Max_Answers_{k}": partial(general_eval, max_pred_answers=k)
    for k in [None,]
}
exact_match_all_eval_funcs = {**max_answers}

wordnet_all_eval_funcs = {
    f"WordNet_{k}": partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}
all_eval_funcs = {}
all_eval_funcs.update(wordnet_all_eval_funcs)

class QuestionAndAnswerClusters(NamedTuple):
    question_id: str
    question: str
    answer_clusters: Dict[frozenset, int]

def load_predict_reuslt(path:str):
    result = OrderedDict()
    if path.endswith('jsonl'):
        info = OrderedDict()
        content = load_data(path, 'jsonl')
        for line in content:
            result.update(line)
            idx = list(line.keys())[0]
            answers = line[idx]
            info[idx] = {'inputs':[''], 'answer': OrderedDict()}
            for ans in answers:
                info[idx]['answer'][ans] = 1
                        
        return result, info
    elif path.endswith('info.json'):
        content = load_data(path, 'json')
        info = OrderedDict()
        for example in content:
            ids = example.get('example_id', '')
            ids = example.get('example_ids', '') if not ids else ids
            answers = list(example['answer'].keys())
            result[ids] = answers
            info[ids] = example

        return result, info



def remove_same(predictions, predictions_info):

    for example_id, prediction in tqdm(predictions.items()):
        info = predictions_info[example_id]
        
        rm_set = set()
        for index, answer in enumerate(prediction):
            if answer in rm_set:
                continue
            cluster = []
            for j, ans in enumerate(prediction[index+1:]):
                cluster.append([[ans,], j+index+1])
            answer_clusters = {
                frozenset(answer_cluster[0]): answer_cluster[1]
                for answer_cluster in cluster
            }
            question = info['inputs'][0]
            
            answers_dict = {example_id: [answer]}
            question_data = {}
            question_data[example_id] = QuestionAndAnswerClusters(
                    question_id=example_id,
                    question=question,
                    answer_clusters=answer_clusters,
                )
            
            if question_data[example_id].answer_clusters:
                eval_func = all_eval_funcs['WordNet_Max_Answers_None']
                scores_None = evaluate(eval_func, question_data, answers_dict=answers_dict)

                for score in scores_None[example_id].score_matrix[0]:
                    if score != 0.0:
                        rm_set.add(prediction[int(score)])

        
        new_prediction = []
        for answer in prediction:
            if answer not in rm_set:
                new_prediction.append(answer)
        for key in list(info['answer'].keys()):
            if key in rm_set:
                del info['answer'][key]
        predictions[example_id] = new_prediction
    return predictions, predictions_info
    

if __name__ == "__main__":
    # predictions_dir = "/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Generation_Model/bart_kepr_4_dpr/bz=1x8x1_ep=1_lr=1e-05_ae=1e-06_seed=220406_keyword_wkdt_multi_4_dpr_rm_bad/evaluate/lastmodel.info.json"
    # predictions_dir = "/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/Finetuned_model/ProtoQA/T5-3B_wkdt_1109/dev-T5.jsonl"
    predictions_dir = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Few-shot_QA/test.questions.v3.jsonl"
    
    
    
    predictions, predictions_info = load_predict_reuslt(predictions_dir)
    predictions, predictions_info = remove_same(predictions, predictions_info)
    
    d = []
    for key, value in predictions.items():
        d.append({key:value[:12]})
    
    dump_data(d, predictions_dir.replace('info.json', '')+'rmsame.jsonl', 'jsonl')