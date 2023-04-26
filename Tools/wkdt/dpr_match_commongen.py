# -*- encoding: utf-8 -*-
'''
@File    :   dpr_match.py
@Time    :   2022/04/18 14:03:56
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
import os
import sys
sys.path.append('../')

from tqdm import tqdm
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from Tools.data_io_util import load_data, dump_data
from Tools.wkdt.match_description import WKDT_matcher

# device = 'cpu'
device = 'cuda:3'

dpr_question_encoder_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/DPR/dpr-question_encoder-single-nq-base"

dpr_context_encoder_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/DPR/dpr-ctx_encoder-single-nq-base"

tokenizer_c = DPRContextEncoderTokenizer.from_pretrained(dpr_context_encoder_path)
model_c = DPRContextEncoder.from_pretrained(dpr_context_encoder_path)
tokenizer_q = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question_encoder_path)
model_q = DPRQuestionEncoder.from_pretrained(dpr_question_encoder_path)
wkdt = WKDT_matcher("/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict",  mode='list')

model_c.to(device)
model_q.to(device)


def DPR_score(query, document):
    with torch.no_grad():
        query_input_ids = tokenizer_q(query, return_tensors='pt')['input_ids']
        query_input_ids = query_input_ids.to(device)
        embeddings_q = model_q(query_input_ids).pooler_output
        
        embeddings_context = []
        for desc in document:
            input_ids = tokenizer_c(desc, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)
            embeddings_c = model_c(input_ids).pooler_output
            embeddings_context.append(embeddings_c)

        scores_list = []
        for embedding in embeddings_context:
            scores_list.append(torch.dot(embedding[0], embeddings_q[0]).item())

    return scores_list

def match_and_select(target, output):
    data = load_data(target, mode='jsonl')
    
    for example in tqdm(data):
        
        concept_list = example['concept_set'].split('#')
        # question = "Make a sentence with '{}': ".format(", ".join(concept_list))    # v1
        question = ", ".join(concept_list)
    
        description_selected = {}
        for word in concept_list:
            # import pdb; pdb.set_trace()
            description_dict = wkdt.match_description(word)
            if description_dict['matched'] != 'MISMATCH':
                description_list = []
                for value in description_dict['desc_list'].values():
                    description_list.extend(value)
                    
                query = f'word "{word}" in sentence "{question}" means what?'
                scores_list = DPR_score(query, description_list)
                
                temp = [(score, desc) for score, desc in zip(scores_list, description_list)]
                temp.sort(reverse = True)
                
                description_selected[word] = [x[1] for x in temp[:2]]
            else:
                description_selected[word] = 'MISMATCH'
    
        example['description_dict'] = description_selected
    
    dump_data(data, output, 'jsonl')
    

if __name__ == "__main__":


    # print(match_and_select_dev('athlete', 'name something that an athelete would not keep in her refrigerator.'))
    
    dataset_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/CommonGen/"
    target = dataset_path + "origin/"
    output = dataset_path + "description_v2/"
    try:
        os.mkdir(output)
    except:
        print(f"{output} exisits.")
    
    # for file in ['train.jsonl', ]:
    for file in ['dev.jsonl', 'test.jsonl', 'train.jsonl']:
        t = os.path.join(target, file)
        o = os.path.join(output, file)
        # load_data(t)
        
        match_and_select(t, o)
