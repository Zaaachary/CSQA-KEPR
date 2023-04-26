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
lemmatizer = WordNetLemmatizer()

def not_the_same(target, gt_list):
    # not subsequence and no cover token
    target_lema_list = []
    for temp in target.split():
        target_lema_list.append(lemmatizer.lemmatize(temp))
    
    for ans in gt_list:
        ans_lema_list = []
        for temp in ans.split():
            ans_lema_list.append(lemmatizer.lemmatize(temp))
        
        for wa in set([target, target.replace(' ',''), *target.split(' '), *target_lema_list]):
            for gt in set([ans, ans.replace(' ',''), *ans.split(' '), *ans_lema_list]):
                if wa in gt or gt in wa:
                    return False
    else:
        return True

def merge_data(*judge_result):
    merge_list = judge_result[0]
    for result_list in judge_result[1:]:
        assert len(result_list) == len(merge_list)
        for m, r in zip(merge_list, result_list):
            # merge_predict
            predict_set = set(m['predict'])
            predict_set.update(r['predict'])
            m['predict'] = list(predict_set)
            # merge right
            m_right = [ans[0] for ans in m['right']]
            for ans in r['right']:
                if ans[0] not in m_right:
                    m['right'].append(ans)
            # merge_wrong
            wrong_set = set(m['wrong'])
            wrong_set.update(r['wrong'])
            m['wrong'] = list(wrong_set)
    return merge_list

def map_weight(predict_list, ground_truth):
    gt_dict = {}
    for item in ground_truth:
        gt_dict[item[0]] = item[1]
    
    for item in predict_list:
        item[1] = gt_dict[item[1]]

def transform_answer_list(answer_list, add_weight=False):
    # remove special token
    if not add_weight:
        for index in range(len(answer_list)):
            answer = answer_list[index]
            answer = answer.replace('\"', '')
            if '/' in answer:
                temp_list = answer.split('/')
                answer = temp_list.pop(0)
                answer_list.extend(temp_list)
            answer_list[index] = answer
    else:
        for index in range(len(answer_list)):
            answer, weight = answer_list[index]
            answer = answer.replace('\"', '')
            if '/' in answer:
                temp_list = answer.split('/')
                answer = temp_list.pop(0)
                answer_list.extend([[item,weight] for item in temp_list])
            answer_list[index][0] = answer
    return answer_list

def all_digit(target_list: List[str]):
    '''judge answers are digit type or not'''
    all_digit_flag = True
    for item in target_list:
        if not item.isdigit():
            all_digit_flag = False
    return all_digit_flag

def clean_protoqa_data(origin_result):
    '''
    return total case and answer set
    '''
    total_answer_set = set()
    total_case = []
    for line in origin_result:
        example_id = line['metadata']['id']
        question = line['question']['normalized']
        answers_list = list(line['answers']['raw'].keys())
        if not all_digit(answers_list):
            total_answer_set.update(answers_list)

        answers_list = [list(item) for item in line['answers']['raw'].items()]
        answers_list = [answer for answer in answers_list if answer[0]!='send us your answers!']
        answers_list.sort(key=lambda x:x[1], reverse=True)
        case = {'id':example_id, 'question':question, 'right_answer': answers_list, 'keyword':line['keyword']}
    
        total_case.append(case)
    return total_case, list(total_answer_set)

def transfer_to_dataset(examples, args):
    total_case = 0
    right_count = 0
    wrong_count = 0
    hard_count = 0
    dataset = []
    cls_dict = {}
    info_dict = {}
    for example in examples:
        example_id = example['id']
        question = example['question']
        
        right_answer_list = example['right_answer']
        wrong_answer_list = example.get('wrong_answer', [])
        if args.dataset_name == "Multi_Classification":
            for ans in right_answer_list:
                total_case += 1
                right_count += 1
                cls_dict[ans[1]] = cls_dict.get(ans[1], 0) + 1
                dataset.append([example_id, question, ans[0], ans[1]])
            for ans in wrong_answer_list:
                wrong_count += 1
                cls_dict[ans[1]] = cls_dict.get(ans[1], 0) + 1
                dataset.append([example_id, question, ans[0], ans[1]])
        elif args.dataset_name == "Binary_Classification":
            for ans in right_answer_list:
                total_case += 1
                right_count += 1
                dataset.append([example_id, question, ans[0], 1, *ans[1:]])
            for ans in wrong_answer_list:
                wrong_count += 1
                dataset.append([example_id, question, ans[0], 0, *ans[1:]])
        elif args.dataset_name == "BC_Hard_Case":
            # for ans
            for ans in right_answer_list:
                total_case += 1
                right_count += 1
                dataset.append([example_id, question, ans[0], 1, ans[1]])
            for ans in wrong_answer_list:
                wrong_count += 1
                if ans[1] in [-1, 0.3]:
                    hard_count += 1
                dataset.append([example_id, question, ans[0], 0, ans[1]])
        elif args.dataset_name in ["Regression", "Satisfy"]:
            for ans in right_answer_list:
                total_case += 1
                dataset.append([example_id, question, *ans])
                
    # print(f'total {len(total_case)} case; right answer {right_count}; wrong answer {wrong_count + hard_wrong_count} (hard {hard_wrong_count})')

    print(f'total {total_case} case; right answer {right_count}; wrong answer {wrong_count}(hard {hard_count})')
    info_dict['total'] = total_case
    info_dict['right'] = right_count
    info_dict['wrong'] = wrong_count
    if args.dataset_name == "Multi_Classification":
        print(cls_dict)

    return dataset, info_dict

def save_param(args, output_top):
    origin = os.path.join(sys.path[0], 'build_dataset.py')
    output = os.path.join(output_top, 'build_dataset.py')
    os.system(f'cp {origin} {output}')

def generate_digit_wrong_case(right_answer_list, num=0):
    digit_list = list(map(int, right_answer_list))
    max_num = max(digit_list)
    min_num = min(digit_list)
    big_flag = True
    num = len(right_answer_list) if num == 0 else num
    for _ in range(num):
        if big_flag:
            wrong = random.randint(int(max_num*1.5), 3*max_num)
            big_flag = False
        else:
            wrong = random.randint(0, int(min_num/2))
            big_flag = True
        yield wrong

def build_Multi_Classification(example_list, answer_list, args):
    if args.dataset_version == 'v1':
        split_method = "weight"
        weight_boundary = [-1, 5, 11, 24, 999]  # -1 add wrong
        wrong_rate = 0.25
    elif args.dataset_version == 'v2':
        split_method = "order"
        # boundary 

    for index_1, example in enumerate(example_list):
        example['right_answer'] = transform_answer_list(example['right_answer'], add_weight=True)

        for index_2, answer in enumerate(example['right_answer']):
            for cls, bucket in enumerate(weight_boundary):
                if answer[1] <= bucket:
                    answer[1] = cls
                    break
        if weight_boundary[0] == -1:
            # add wrong
            example['wrong_answer'] = []
            right_answer_list = [item[0] for item in example['right_answer']]
            if all_digit(right_answer_list):
                gen = generate_digit_wrong_case(right_answer_list)
                while len(example['wrong_answer']) < len(right_answer_list) * wrong_rate:
                    wrong_digit = next(gen)
                    if wrong_digit not in example['wrong_answer']:
                        example['wrong_answer'].append([wrong_digit, 0])
            else:
                while len(example['wrong_answer']) < len(right_answer_list) * wrong_rate:
                    wrong_ans = random.choice(answer_list)
                    if wrong_ans not in example['wrong_answer'] and wrong_ans not in example['right_answer']:
                        example['wrong_answer'].append([wrong_ans, 0])
    return example_list

def build_Binary_Classification(example_list, answer_list, args):
    if args.dataset_version == 'v1':
        # positive case -> -> 0.6 ~ 0.90 weight(top1 anchor)
        # negative caas -> 0.1 +- 0.05
        no_zero, no_overf = True, True
        threshold = 0.85
        top1_anchor = True

    new_example_list = []
    for index_1, example in enumerate(example_list):

        weight_list = [ans[1] for ans in example['right_answer']]
        weight_sum = sum(weight_list)
        if 0 in weight_list and no_zero:
            continue
        elif no_overf and weight_sum > 100:
            continue
        elif weight_sum < threshold:
            continue

        right_answer_list = [item[0] for item in example['right_answer']]
        
        example['right_answer'] = transform_answer_list(example['right_answer'], add_weight=True)
        example['right_answer'].sort(key=lambda x:x[1], reverse=True)

        # top1_anchor
        top1_score = example['right_answer'][0][1]
        for answer in example['right_answer']:
            label = round(answer[1] / top1_score * 0.3 + 0.6, 3) 
            answer.insert(1, label)

        example['wrong_answer'] = []
        digit = all_digit(right_answer_list)
        if digit:
            gen = generate_digit_wrong_case(right_answer_list)
            while len(example['wrong_answer']) < len(example['right_answer']):
                label = round(random.random() * 0.1 + 0.05, 3)
                wrong_digit = next(gen)
                if wrong_digit not in example['wrong_answer']:
                    example['wrong_answer'].append([str(wrong_digit), label, -2])
        else:
            while len(example['wrong_answer']) < len(example['right_answer']):
                label = round(random.random() * 0.1 + 0.05, 3)
                wrong_ans = random.choice(answer_list)
                if wrong_ans not in example['wrong_answer'] and wrong_ans not in example['right_answer']:
                    example['wrong_answer'].append([wrong_ans, label, -2])

        new_example_list.append(example)
    return new_example_list

def build_BC_Hard_Case(example_list, generated_list, answer_list, args):
    '''
    right: ans, weight
    wrong: ans, 0/-1   0: random sample, -1 hard case
    '''
    new_example_list = []
    no_gen_small = False
    top = False
    if args.dataset_version == 'v1':
        # gt: 过多答案的进行抽样；使用新的 hard case
        # gt best * 2; gen right *1
        # gt 较少的样本不引入生成的负例只使用random sample
        # gt 较多的样本，使用 top2 answer + 生成的最后一个正确答案
        transfer_answer_list = False
        gt_case_limit = 7
        gen_small_gt = 3    # threshold for adding wrong gen
        gen_ratio = 0.5
        nodigit = True
        gen_gt = True
    elif args.dataset_version == 'v2':
        transfer_answer_list = False
        gt_case_limit = 7
        gen_small_gt = 3    # threshold for adding wrong gen
        gen_ratio = 0.5
        gen_gt = False
        nodigit = True
    elif args.dataset_version == 'v2.1_fix_soft':
        # 减少了重复词项
        transfer_answer_list = False
        gt_case_limit = 6
        gen_small_gt = 4
        no_gen_small = True
        gen_ratio = 0.8
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3':
        top = True
        transfer_answer_list = False
        gt_case_limit = 2
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 1
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3.1':
        top = True
        transfer_answer_list = False
        gt_case_limit = 2
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 0
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3.2':
        top = True
        transfer_answer_list = False
        gt_case_limit = 2
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 0.5
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3.3':
        top = True
        transfer_answer_list = False
        gt_case_limit = 3
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 0
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3.4':
        top = True
        transfer_answer_list = False
        gt_case_limit = 4
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 0
        nodigit = True
        gen_gt = False
    elif args.dataset_version == 'v3.56':
        top = True
        transfer_answer_list = False
        gt_case_limit = 5
        gen_small_gt = 3
        no_gen_small = True
        gen_ratio = 0
        nodigit = True
        gen_gt = False
    
    count_ge = {}
    count_gt = {}
    
    for index_1, (example, gen_example) in enumerate(zip(example_list, generated_list)):
        # preprocess map weight
        # map_weight(gen_example['right'], example['right_answer'])
        gen_answers = gen_example['answers']
        gen_right_answer = []
        gen_wrong_answer = []
        
        new_right_answer = []
        
        for key, value in gen_answers.items():
            if value > 0:
                gen_right_answer.append([key, int(value)])
            else:
                gen_wrong_answer.append(key)
        
        if transfer_answer_list:
            example['right_answer'] = transform_answer_list(example['right_answer'], add_weight=True)
        
        # all right answer in current example
        gt_right_answer_list = [item[0] for item in example['right_answer']]
        gen_right_answer_list = [item[0] for item in gen_right_answer]
        merge_right_answer_list = list(set(gt_right_answer_list+gen_right_answer_list))

        # score -> soft
        example['right_answer'].sort(key=lambda x:x[1], reverse=True)
        top1_score = example['right_answer'][0][1]
        for answer in example['right_answer']:
            label = round(answer[1] / top1_score * 0.1 + 0.8, 3) 
            answer[1] = label

        # avoid too much answers
        # import pdb; pdb.set_trace()
        if len(gt_right_answer_list) > gt_case_limit:
            if top:
                new_right_answer.extend(example['right_answer'][:gt_case_limit])
            else:
                # ensure the high score answer; sample low score answer
                new_right_answer.append(example['right_answer'][0])
                temp = random.sample(example['right_answer'][1:], gt_case_limit-1)
                temp.sort(key=lambda x:x[1], reverse=True)
                new_right_answer.extend(temp)
        else:
            new_right_answer = example['right_answer']
        
        if gen_gt:
        # add generated answer
            if len(new_right_answer) < gt_case_limit and len(merge_right_answer_list) > len(gt_right_answer_list):
                for answer in gen_right_answer:
                    if answer[0] not in gt_right_answer_list:
                        new_right_answer.append(answer)
                    if len(new_right_answer) == gt_case_limit:
                        break
        
        gt_len = len(gt_right_answer_list)
        ge_len = len(gen_right_answer)
        count_gt[gt_len] = count_gt.get(gt_len, 0) + 1
        count_ge[ge_len] = count_ge.get(ge_len, 0) + 1
        
        # organize wrong answer
        new_wrong_answer = []
        digit = all_digit(gt_right_answer_list)
        if digit:
            if nodigit:
                continue
            gen = generate_digit_wrong_case(gt_right_answer_list, num=len(example['right_answer']))
            while len(new_wrong_answer) < len(new_right_answer):
                wrong_digit = next(gen)
                if wrong_digit not in new_wrong_answer:
                    new_wrong_answer.append([wrong_digit, 0.1])
        else:
            if len(gt_right_answer_list) > gen_small_gt and gen_wrong_answer:
                target_gen = gen_ratio * len(new_right_answer)
                while len(new_wrong_answer) < int(target_gen) and gen_wrong_answer:
                    wa = random.choice(gen_wrong_answer)
                    del gen_wrong_answer[gen_wrong_answer.index(wa)]
                    if not_the_same(wa, merge_right_answer_list + new_wrong_answer):
                        new_wrong_answer.append(wa)

                new_wrong_answer = [[answer, 0.3] for answer in new_wrong_answer]
            elif len(gt_right_answer_list) < gen_small_gt and no_gen_small:
                continue
            # random sample
            while len(new_wrong_answer) < len(new_right_answer):
                wrong_ans = random.choice(answer_list)
                if wrong_ans not in [ans[0] for ans in new_wrong_answer] \
                    and wrong_ans not in merge_right_answer_list:
                    new_wrong_answer.append([wrong_ans, 0.1])

        example['right_answer'] = new_right_answer
        example['wrong_answer'] = new_wrong_answer
        new_example_list.append(example)
        
    count_gt = [(key, value) for key, value in count_gt.items()]
    count_gt.sort(key=lambda x:x[0])
    count_ge = [(key, value) for key, value in count_ge.items()]
    count_ge.sort(key=lambda x:x[0])
    print(count_gt)
    print(count_ge)
    return new_example_list

def build_Regression(example_list, args):
    if args.dataset_version == 'v1':
        # use weight sum >= 85 example, no zero & no over
        # scaling to 100; no ans transform
        threshold = 85
        no_zero, no_overf = True, True
    elif args.dataset_version == 'v2':
        threshold = 85
        no_zero, no_overf = True, True
        top1_anchor = True

    new_example_list = []

    for index_1, example in enumerate(example_list):
        weight_list = [ans[1] for ans in example['right_answer']]
        weight_sum = sum(weight_list)
        if 0 in weight_list and no_zero:
            continue
        elif no_overf and weight_sum > 100:
            continue
        elif weight_sum < threshold:
            continue

        example['right_answer'].sort(key=lambda x:x[-1], reverse=True)
        if top1_anchor:
            top1_score = example['right_answer'][0][1]
            for answer in example['right_answer']:
                answer.append(round(answer[1] / top1_score * 10, 3))
            
        else:
            scale = 100 / weight_sum
            for answer in example['right_answer']:
                answer.append(round(answer[1] * scale * 0.1, 4))

        new_example_list.append(example)
        
    return new_example_list

def build_Satisfy(example_list, generated_list, answer_list, args):
    if args.dataset_version == 'v1':
        # ground truth only
        no_zero, no_overf = True, True

    del_count = 0

    new_example_list = []
    for index_1, example in enumerate(example_list):

        # clean answer
        del_index = []
        for index_a, answer in enumerate(example['right_answer']):
            if no_zero and answer[1] <= 0:
                del_index.append(index_a)
            elif no_overf and answer[1] >= 100:
                del_index.append(index_a)
            answer[1] = round(answer[1] * 0.01, 2)

        del_count += len(del_index)
        for index_d in del_index[::-1]:
            del example['right_answer'][index_d]

        # right_answer_list = [item[0] for item in example['right_answer']]
        
        example['right_answer'] = transform_answer_list(example['right_answer'], add_weight=True)
        example['right_answer'].sort(key=lambda x:x[1], reverse=True)

        if len(example['right_answer']) < 2:
            continue
        new_example_list.append(example)
    return new_example_list

def main(args):
    output_top = os.path.join(args.output_path, args.dataset_name)
    if not os.path.exists(output_top):
        os.mkdir(output_top)
    output_top = os.path.join(output_top, args.dataset_version)
    if not os.path.exists(output_top):
        os.mkdir(output_top)
    elif not args.test:
        print('exists')
        return 

    for datatype in ['dev', 'train']:
        origin_path = os.path.join(args.origin_dataset, f"{datatype}.scraped.jsonl")
        output_path = os.path.join(output_top, f"{datatype}.tsf")
        if args.generated_result:
            generated_path_list = [os.path.join(path, f"{datatype}.info.judge.jsonl")for path in args.generated_result]
        info_path = os.path.join(output_top, f"{datatype}.info.json")
    
        origin_data = load_data(origin_path, 'jsonl')
        example_list, answer_list = clean_protoqa_data(origin_data)
        if args.dataset_name == "Multi_Classification":
            data_list = build_Multi_Classification(example_list, answer_list, args)        
            data_list, info_dict = transfer_to_dataset(data_list, args)
        elif args.dataset_name == "Binary_Classification":
            data_list = build_Binary_Classification(example_list, answer_list, args)
            data_list, info_dict = transfer_to_dataset(data_list, args)
        elif args.dataset_name == "BC_Hard_Case":
            judged_data = []
            for path in generated_path_list:
                data = load_data(path, 'jsonl')
                judged_data.append(data)
            if len(judged_data) >= 2:
                generated_list = merge_data(*judged_data)
            else:
                generated_list = judged_data[0]
            data_list = build_BC_Hard_Case(example_list, generated_list, answer_list, args)
            data_list, info_dict = transfer_to_dataset(data_list, args)
        elif args.dataset_name == "Regression":
            data_list = build_Regression(example_list, args)
            data_list, info_dict = transfer_to_dataset(data_list, args)
        elif args.dataset_name == "Satisfy":
            data_list = build_Satisfy(example_list, [], answer_list, args)
            data_list, info_dict = transfer_to_dataset(data_list, args)

        dump_data(data_list, output_path, 'tsf')
        dump_data(info_dict, info_path, 'json')

    save_param(args, output_top)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_dataset", type=str, required=True)
    parser.add_argument("--generated_result", nargs='*')
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_name", required=True,
        choices=['Multi_Classification', 'Binary_Classification', 'BC_Hard_Case', 'Regression', 'Satisfy'])
    parser.add_argument("--dataset_version", type=str, required=True)
    parser.add_argument("--test", action='store_true')
    # --test
    # --test
    # --test
    args_str = r"""
    --origin_dataset /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1
    --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/
    --generated_result /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/ALL_GENERATEDRAW/bart-normal
    --dataset_name BC_Hard_Case
    --dataset_version v3.56
    """
    args = parser.parse_args(args_str.split())
    print(args)
    random.seed(42)
    main(args)