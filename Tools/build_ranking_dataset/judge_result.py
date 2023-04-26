# -*- encoding: utf-8 -*-
'''
@File    :   result_cover.py
@Time    :   2021/12/29 16:17:52
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   compute the coverage of ground truth in preditions
'''
import json

from tqdm import tqdm

from protoqa_evaluator.scoring import wordnet_score


def load_ground_truth(path):
    f = open(path, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()
    result = []
    for line in content:
        case = json.loads(line)
        question = case['question']['original']
        answers_list = list(case['answers']['raw'].keys())
        
        temp = {'question': question, "answers": answers_list}
        result.append(temp)

    return result

def load_predict_reuslt(path) -> list:
    f = open(path, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()
    result = []
    for line in content:
        case = json.loads(line)
        result.append(case)
    
    return result

def dump_judge_result(path, predictions):
    f = open(path, 'w', encoding='utf-8')
    for case in predictions:
        f.write(json.dumps(case)+'\n')
    f.close()


def match_answer(prediction, answer_list) -> bool :
    total_score = 0
    max_score = 0
    matched_answer = ''
    for answer in answer_list:
        # import pdb; pdb.set_trace()
        # try:
        score = wordnet_score(prediction, answer)
        # except:
            # print(prediction, answer)
        if score > max_score: 
            max_score = score
            matched_answer = answer  
    if max_score > 0:
        # if matched_answer != prediction:
        #     print(max_score,matched_answer,prediction)
        return True, matched_answer
    else:
        return False, None

def judgue_result(predictions, ground_truth, add_matched_answer):
    total_wrong = 0
    total_right = 0
    total_answer = 0

    for index, (prediction_dict, ground_truth_dict) in enumerate(tqdm(list(zip(predictions, ground_truth)))):
        right, wrong = [], []
        case_id, predictions_list = prediction_dict.popitem()

        temp_dict = {}

        question = ground_truth_dict['question']
        ground_truth_list = ground_truth_dict['answers']
        total_answer += len(ground_truth_dict)

        for ans in predictions_list:
            ans = ' '.join(ans.split(' ')[:4])

            mc, mc_ans = match_answer(ans, ground_truth_list)
            if mc:
                total_right += 1
                if add_matched_answer:
                    right.append((ans, mc_ans))
                else:
                    right.append(ans)
            else:
                total_wrong += 1
                wrong.append(ans)
        
        if add_matched_answer:
            covered = []
            uncovered = []
            for ans in ground_truth_list:
                for a in right:
                    if ans in a[1]:
                        covered.append(ans)
                        break
                else:
                    uncovered.append(ans)
                    
        
        temp_dict['id'] = case_id
        # temp_dict['question'] = question
        temp_dict['predict'] = predictions_list
        temp_dict['right'] = right
        temp_dict['wrong'] = wrong
        temp_dict['covered'] = covered
        temp_dict['uncovered'] = uncovered
        # temp_dict['answers'] = ground_truth_list

        predictions[index] = temp_dict
    print(total_wrong, total_right, total_answer)
    return predictions

def judgue_result_for_check(predictions, ground_truth):
    info_merge = []

    for index in tqdm(range(len(predictions))):
        prediction_dict, gt_dict = predictions[index], ground_truth[index]
        import pdb; pdb.set_trace()

    return 



if __name__ == "__main__":

    add_matched_answer = True
    
    # predictions = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/ALL_GENERATEDRAW/well-trained-BART-300-10/train.jsonl"
    predictions = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/ALL_GENERATEDRAW/bart-normal/dev.jsonl"
    # ground_truth = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/training/train.scraped.jsonl"
    ground_truth = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/training/dev.scraped.jsonl"
    # ground_truth = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/dev.crowdsourced.jsonl"

    output = predictions + '.judge'
    # output = "/data1/zhifli/Models/proto-qa/gpt2_baseline/model-5_32/step=11256.sample-3.judge.jsonl"

    # predictions = "/data1/zhifli/Models/proto-qa/gpt2_baseline/model-2_32_6/prediction_300_topk_0.jsonl"
    # ground_truth = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/evaluate/dev.questions.jsonl"
    # output = "/data1/zhifli/Models/proto-qa/gpt2_baseline/model-2_32_6/sample4-count.jsonl"

    ground_truth_list = load_ground_truth(ground_truth)

    predictions_list = load_predict_reuslt(predictions)

    predictions_list = judgue_result(predictions_list, ground_truth_list, add_matched_answer)

    dump_judge_result(output, predictions_list)