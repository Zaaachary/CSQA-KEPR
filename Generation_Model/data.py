import os
import time
import json
import logging
import random
from multiprocessing import Pool, cpu_count     # https://docs.python.org/3/library/multiprocessing.html
from collections import OrderedDict
from itertools import chain
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader, sampler
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import sys
sys.path.append('../')
from Tools.wkdt.match_description import WKDT_matcher
from Tools.data_io_util import load_data

class ProtoQA_Dataset(Dataset):

    def __init__(self, 
        dataset_path, model_type='gpt2',
        max_src_len=None, max_tgt_len=None, 
        overwrite_cache=False, tokenizer=None, 
        dataset_type='train', evaluate=False,
        wkdt_path='', experiment='', seed=42
        ):

        super().__init__()
        self.evaluate = evaluate
        self.raw_examples = []
        self.labels = []
        self.examples = []
        self.scores = []

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.experiment = experiment
        self.model_type = model_type
        self.seed = seed

        if 'keyword_wkdt' in self.experiment:
            self.wkdt = WKDT_matcher(wkdt_path, experiment=self.experiment)

        if evaluate:
            self.load_data()
        else:
            file_name = f"{model_type}_cached_lm_{dataset_type}.pkl"

            cached_feature_file = os.path.join(
                self.dataset_path, file_name
            )
            if os.path.exists(cached_feature_file) and not overwrite_cache:
                logging.info(f"loading features from {file_name}")
                with open(cached_feature_file, 'rb') as handle:
                    self.raw_examples = pickle.load(handle)
                    self.examples = pickle.load(handle)
            else:
                self.load_data()
                self.convert_tokens_to_ids()

                # logging.info(f"dumping features to {file_name}")
                # with open(cached_feature_file, 'wb') as handle:
                    # pickle.dump(self.raw_examples, handle)
                    # pickle.dump(self.examples, handle)

    def load_data(self):
        if not os.path.isfile(self.dataset_path):
            if not self.evaluate:
                # no need label
                target = os.path.join(self.dataset_path, f"{self.dataset_type}.scraped.jsonl")
            else:
                target = os.path.join(self.dataset_path, f"{self.dataset_type}.questions.jsonl")
        else:
            target = self.dataset_path
        assert os.path.isfile(target)
        
        f = open(target, 'r', encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            # try:
            line_dict = json.loads(line)
            # except:
                # import pdb; pdb.set_trace()
                # print(1)

            example_id = line_dict['metadata']['id']
            question = line_dict['question']['original']

            question = self.transform_question(question)
            # import pdb; pdb.set_trace()
            if 'keyword_wkdt' in self.experiment:
                keyword = line_dict['keyword']
                description = []
                for word in keyword:
                    if 'dpr' in self.experiment:
                        desc = line_dict['description_dict'][word]
                        if desc != 'MISMATCH':
                            des = f"{word}: {desc}"
                            des = des + '.' if des[-1] != '.' else des
                            description.append(des)
                    else:
                        wkdt = self.wkdt.match_description(word)
                        if wkdt['matched'] != "MISMATCH":
                            des = f"{word}: {wkdt['desc_list'][0]}"
                            des = des + '.' if des[-1] != '.' else des
                            description.append(des)

            if self.evaluate:
                temp = [example_id, question]
                if 'keyword_wkdt' in self.experiment:
                    if 'multi_3' in self.experiment:
                        temp.append('; '.join(description[:3] if description else ''))
                    elif 'multi_4' in self.experiment:
                        temp.append('; '.join(description[:4] if description else ''))
                    elif 'multi' in self.experiment:
                        temp.append('; '.join(description[:2] if description else ''))
                    elif 'tp2' in self.experiment and len(description) >=2:
                        temp.append(description[1] if description else '')
                    else:
                        temp.append(description[0] if description else '')
                self.raw_examples.append(temp)
            else:
                if 'order' in self.experiment:
                    temp = line_dict['answers']['raw']
                    answers = list(temp.items())
                    answers.sort(key=lambda x:x[1], reverse=True)
                    # scores = [answer[1] for answer in answers]
                    answers = [answer[0] for answer in answers]
                    scores = [0] * len(answers)
                    if len(scores) >= 2:
                        scores[:2] = [1,1]
                    self.scores.extend(scores)
                else:
                    answers = list(line_dict['answers']['raw'].keys())
                
                # ans transform
                if 'order' in self.experiment:
                    answers, scores = self.transform_answer_list(answers, scores)
                else:
                    answers = self.transform_answer_list(answers)
                
                if 'rm_bad' in self.experiment:
                    answers = [answer for answer in answers if answer != 'send us your answers!']
                                    
                answers = [' '+answer+'.' for answer in answers]    # add space
                
                for answer in answers:
                    temp = [question, answer]
                    if 'keyword_wkdt' in self.experiment:
                        if 'multi_3' in self.experiment:
                            temp.append('; '.join(description[:3] if description else ''))
                        elif 'multi_4' in self.experiment:
                            temp.append('; '.join(description[:4] if description else ''))
                        elif 'multi' in self.experiment:
                            temp.append('; '.join(description[:2] if description else ''))
                        else:
                            temp.append(description[0] if description else '')
                    
                    self.raw_examples.append(temp)
        logging.info(f"{self.dataset_type} dataset loaded")

    class Convert:
        # tokenize and get the max len   for multiprocessing
        def __init__(self, 
            tokenizer, experiment, 
            evaluate, model_type='gpt2',
            max_src_len=50, max_tgt_len=50
            # max_seq_len=None      # TODO
            ):
            self.tokenizer = tokenizer
            self.evaluate = evaluate
            self.experiment = experiment
            self.model_type = model_type
            self.max_src_len = max_src_len   # for bart encoder
            self.max_tgt_len = max_tgt_len   # for gpt2 & bart decoder

        def __call__(self, raw_example):
            if self.model_type.lower() in ['bart', 't5']:
                try:
                    return self._bart(raw_example)
                except:
                    logging.warning(raw_example)
                    exit()
            elif self.model_type == 'gpt2':
                return self._gpt2(raw_example)

        def _bart(self, raw_example):
            
            if 'keyword_wkdt' in self.experiment:
                question, answer, description = raw_example
                source_inputs = self.tokenizer(description, question + self.tokenizer.mask_token, 
                    add_prefix_space=True, max_length=self.max_src_len, padding='max_length', truncation='only_first')
                if "enc_only" in self.experiment:
                    target_inputs = self.tokenizer.encode(question + answer, 
                        add_prefix_space=True, max_length=self.max_tgt_len, padding='max_length', truncation='only_first')
                else:
                    target_inputs = self.tokenizer.encode(description, question + answer, 
                        add_prefix_space=True, max_length=self.max_tgt_len, padding='max_length', truncation='only_first')
            else:
                question, answer = raw_example
                source_inputs = self.tokenizer(question + self.tokenizer.mask_token, 
                    add_prefix_space=True, max_length=self.max_src_len, padding='max_length', truncation=True)
                target_inputs = self.tokenizer.encode(question + answer, 
                    add_prefix_space=True, max_length=self.max_tgt_len, padding='max_length', truncation=True)

            feature_dict = source_inputs
            feature_dict['decoder_input_ids'] = target_inputs
            return feature_dict

        def _gpt2(self, raw_example):
            if "keyword_wkdt" in self.experiment:
                question, answer, description = raw_example
                tokenized_source = self.tokenizer.encode(description+self.tokenizer.eos_token, question, max_length=self.max_src_len, truncation='only_first')
                tokenized_answer = self.tokenizer.encode(answer+self.tokenizer.eos_token)
                return (tokenized_source, tokenized_answer)
            else:
                question, answer = raw_example
                tokenized_question = self.tokenizer.encode(question)
                tokenized_answer = self.tokenizer.encode(answer+self.tokenizer.eos_token)
                return (tokenized_question, tokenized_answer)
    
    def convert_tokens_to_ids(self):
        '''
        make input and label
        tokenized_examples: 
        -gpt2 [question, answer] 
        -bart [
            source: <s>question <mask></s>
            target: <s>question answer.</s>
            ]
        '''
        
        logging.info(f"tokenizing {self.dataset_type} examples")
        logging.info(f'data format {self.raw_examples[0]}')
        
        # import pdb; pdb.set_trace()
        
        now = time.time()
        if not self.tokenizer.mask_token:
            self.tokenizer.mask_token = '<extra_id_0>'
        with Pool(processes=min(8, cpu_count())) as pool:
            tokenized_examples = pool.map(
                self.Convert(self.tokenizer, self.experiment,
                self.evaluate, self.model_type, self.max_src_len, self.max_tgt_len),
                self.raw_examples)
        logging.info(f"start {min(8, cpu_count())} processes, cost {time.time() - now}")

        if not self.max_src_len and self.model_type == 'gpt2':
            self.max_src_len = max([len(example[0]) + len(example[1]) for example in tokenized_examples])
            logging.info(f"GPT2 max_seq_len {self.max_src_len}")

        if self.evaluate:
            # TODO
            for example in tokenized_examples:
                q = example
                self.examples.append(q)
        else:
            if self.model_type == 'gpt2':
                # add eos token or tuncation
                for example in tokenized_examples:
                    q, a = example
                    total_len = len(q) + len(a)
                    if total_len < self.max_src_len:
                        for _ in range(total_len, self.max_src_len):
                            a.append(self.tokenizer.eos_token_id)
                    elif total_len > self.max_src_len:
                        longer = total_len - self.max_src_len
                        q = q[longer:]      # cut head
                    
                    qa = list(chain(q, a))
                    self.examples.append(qa)
                    # make label
                    label = qa[:]
                    label[:len(q)] = [-100] * len(q)
                    end = False
                    for index in range(len(q), len(label)):
                        if label[index] == self.tokenizer.eos_token_id:
                            if not end:
                                end = True
                            else:
                                label[index] = -100
                    self.labels.append(label)
            elif self.model_type.lower() in ['bart', 't5']:
                for feature_dict in tokenized_examples:
                    source_input_ids = feature_dict['input_ids']
                    label = feature_dict['decoder_input_ids'][1:]
                    decoder_input_ids = feature_dict['decoder_input_ids'][:-1]# no need to input the last token 
                    feature_dict['decoder_input_ids'] = decoder_input_ids
                    self.examples.append(feature_dict)

                    if 'enc_only' in self.experiment:
                        question_start = source_input_ids.index(2) + 2
                        q_part = True
                        for index, token in enumerate(label):
                            if q_part:
                                if token == source_input_ids[question_start+index]:
                                    label[index] = -100
                                else:
                                    q_part = False
                            elif token == self.tokenizer.pad_token_id:
                                label[index] = -100
                    else:
                        # mask question part
                        q_part = True
                        for index, token in enumerate(label):
                            if q_part:
                                if token == source_input_ids[index+1]:
                                    label[index] = -100
                                else:
                                    q_part = False
                            elif token == self.tokenizer.pad_token_id:
                                label[index] = -100
                            elif token == self.tokenizer.eos_token_id and 'rank' in self.experiment:
                                label[index] = -100
                    self.labels.append(label)

    def order(self):
        zero_group = [[score, example, label] for score, example, label in zip(self.scores, self.examples, self.labels) if score == 0]
        one_group = [[score, example, label] for score, example, label in zip(self.scores, self.examples, self.labels) if score == 1]
        merge_group = one_group + zero_group
        random.seed(self.seed)
        random.shuffle(zero_group)
        random.shuffle(one_group)
        random.shuffle(merge_group)
        self.scores = []
        self.examples = []
        self.labels = []
        if 'order0' in self.experiment:
            target = one_group + zero_group
        elif 'order1' in self.experiment:
            target = zero_group + one_group 
        elif 'order2' in self.experiment:
            target = merge_group + one_group
        elif 'order3' in self.experiment:
            half = int(len(one_group)/2)
            front, end = one_group[:half], one_group[half:]
            target = front + zero_group + end
        elif 'order4' in self.experiment:
            half = int(len(one_group)/2)
            front, end = one_group[:half], one_group[half:]
            merge2 = zero_group + front
            random.shuffle(merge2)
            target = merge2 + end
        
        for score, example, label in target:
            self.scores.append(score)
            self.examples.append(example)
            self.labels.append(label)

    def make_dataloader(self, batch_size):
        if self.dataset_type == "train":
            data_sampler = RandomSampler(self)
        else:
            data_sampler = SequentialSampler(self)

        if 'order' in self.experiment:
            if self.dataset_type == "train":
                self.order()
            data_sampler = SequentialSampler(self)

        if self.model_type == 'gpt2':
            dataloader = DataLoader(self, sampler=data_sampler, batch_size=batch_size, num_workers=4) # TODO
        elif self.model_type.lower() in ['bart', 't5']:
            dataloader = DataLoader(self, sampler=data_sampler, batch_size=batch_size, num_workers=4, collate_fn=self.collate_fn) # TODO

        return dataloader

    @staticmethod
    def transform_answer_list(answer_list, scores=None):
        
        for index in range(len(answer_list)):
            answer = answer_list[index]
            answer = answer.replace('\"', '')
            if '/' in answer:
                temp_list = answer.split('/')
                answer = temp_list.pop(0)
                if scores:
                    score = scores[index]
                    scores.extend([score] * len(temp_list))
                answer_list.extend(temp_list)
                
            answer_list[index] = answer
        if scores:
            return answer_list, scores
        else:
            return answer_list

    @staticmethod
    def transform_question(origin):
        '''
        > after having kids name something that happens that interrupts a couples alone time at night

        > after having kids one thing that happens that interrupts a couples alone time at night is

        '''
        question = origin.lower()
        question = question.replace('.', '')
        question = question.replace(':', '')
        question = question.replace('?', '')
        question = question.replace('someone', 'one person')
        question = question.replace('someplace', 'one place')
        transform_dict = {
            "name something": "one thing",
            'tell me something': 'one thing',
            'name a ': 'one ',
            "name an ": "one ",
            "name": "",
            # "name ": "",
            # "name another ": "another ",
            "SW tell me a ": "one ",
            "SW tell me an ": "one ",
            "SW what": "one",
            "SW give me a ": "one ",
            "SW tell me ": "",
            "which": "one",
            "what": "one",
            "how can you tell": "one way to tell",
        }
        order = ['name something', 'tell me something', 'name a ', 'name an ', 'name',
            'SW tell me a ', 'SW tell me an ', 'SW what', 'SW give me a ', 'SW tell me ',
            'which', 'what', 'how can you tell']
        transform = OrderedDict.fromkeys(order)
        transform.update(transform_dict)

        for pattern, trans in transform.items():
            if pattern.startswith('SW') and pattern[3:] in question:
                question = question.replace(pattern[3:], trans)
                question = question.strip() + ' is'
                break
            elif pattern in question:
                question = question.replace(pattern, trans)
                question = question.strip() + ' is'
                break
        else:
            question = 'Q: ' + question +'? A: '

        question = question[0].upper() + question[1:]

        return question

    @staticmethod
    def transform_question_for_list(origin):
        question = origin.lower().strip()
        question = 'Q: ' + question +' A: '

        question = question[0].upper() + question[1:]

        return question

    def collate_fn(self, batch):
        # TODO  add GPT2
        if self.model_type == 'gpt2':
            pad_token_id = self.tokenizer.eos_token_id
            max_inputids_len = 0
            for example in batch:
                input_ids = example['input_ids']
                for index, token in enumerate(input_ids):
                    if token == pad_token_id:
                        max_inputids_len = max(max_inputids_len, index+1)
                        break

            input_ids = []
            labels = []

            for example in batch:
                input_ids.append(example['input_ids'][:max_inputids_len])
                labels.append(example['labels'][:max_dec_inputids_len])
            
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
                    
            batch = {
                "input_ids": input_ids,
                "labels": labels
            }
            return batch
        else:
            pad_token_id = self.tokenizer.pad_token_id
            max_inputids_len = 0
            max_dec_inputids_len = 0
            for example in batch:
                input_ids = example['input_ids']
                decoder_input_ids = example['decoder_input_ids']
                for index, token in enumerate(input_ids):
                    if token == pad_token_id:
                        max_inputids_len = max(max_inputids_len, index)
                        break
                for index, token in enumerate(decoder_input_ids):
                    if token == pad_token_id:
                        max_dec_inputids_len = max(max_dec_inputids_len, index)
                        break
            input_ids = []
            masks = []
            target_ids = []
            labels = []
            for example in batch:
                input_ids.append(example['input_ids'][:max_inputids_len])
                masks.append(example['attention_mask'][:max_inputids_len])
                target_ids.append(example["decoder_input_ids"][:max_dec_inputids_len])
                labels.append(example['labels'][:max_dec_inputids_len])
            
            input_ids = torch.stack(input_ids)
            masks = torch.stack(masks)
            target_ids = torch.stack(target_ids)
            labels = torch.stack(labels)
                    
            batch = {
                "input_ids": input_ids,
                "attention_mask": masks,
                "decoder_input_ids": target_ids,
                "labels": labels
            }
            return batch

    def __len__(self):
        if self.evaluate:
            return len(self.raw_examples)
        else:
            return len(self.examples)

    def __getitem__(self, idx):
        if self.evaluate:
            return self.raw_examples[idx]
            # return {'input_ids': torch.tensor(self.examples[idx])}
        else:
            if self.model_type.lower() in ['bart', 't5']:
                feature_dict= {}
                for key, value in self.examples[idx].items():
                    feature_dict[key] = torch.tensor(value)
                feature_dict['labels'] = torch.tensor(self.labels[idx])
                return feature_dict
            elif self.model_type == 'gpt2':
                return {
                    'input_ids': torch.tensor(self.examples[idx]),
                    'labels': torch.tensor(self.labels[idx])
                }


class CommonGen_Dataset(Dataset):

    def __init__(self, 
        dataset_path, max_src_len=None, max_tgt_len=None, 
        tokenizer=None, dataset_type='train', evaluate=False,
        experiment=''
        ):

        self.evaluate = evaluate
        self.raw_examples = []
        self.labels = []
        self.examples = []
        self.scores = []
        self.example_ids = []

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        self.experiment=experiment
        wkdt_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict"
        if 'ke' in self.experiment:
            self.wkdt = WKDT_matcher(wkdt_path, experiment=self.experiment)


        self.load_data()
        if not evaluate:
            self.convert_tokens_to_ids()

    def load_data(self):
        if os.path.isfile(self.dataset_path):
            target = self.dataset_path
        else:
            target = os.path.join(self.dataset_path, f"{self.dataset_type}.jsonl")
        data_list = load_data(target, mode='jsonl')
        
        
        
        if not self.evaluate:
            for data in data_list:
                concept_set = data['concept_set'].split('#')
                scene = data['scene']
                example = [', '.join(concept_set), ]
                if 'ke' in self.experiment:
                    description = ''
                    if 'dpr' in self.experiment:
                        # import pdb; pdb.set_trace()
                        description_dict = data['description_dict']
                        for concept in concept_set:
                            temp_list = description_dict[concept] if description_dict[concept] != 'MISMATCH' else ' '
                            temp = temp_list[0]
                            temp = temp + '.' if temp[-1] != '.' else temp
                            description += f'{concept.lower()}: {temp.lower()}; '
                    else:
                        for concept in concept_set:
                            wkdt = self.wkdt.match_description(concept)
                            # import pdb; pdb.set_trace()
                            if wkdt['matched'] != "MISMATCH":
                                des = f"{concept}: {wkdt['desc_list'][0]}"
                                des = des + '.' if des[-1] != '.' else des
                                description += des.lower() + '; '
                    if 'proto' in self.experiment:
                        description += '' + data['prototype'][0]
                    example.insert(0, description)
                else:
                    if 'proto' in self.experiment:
                        example.insert(0, data['prototype'][0])

                for sc in scene:
                    self.raw_examples.append(example+[sc])
        else:
            for index, data in enumerate(data_list):
                concept_set = data['concept_set'].split('#')
                example = [', '.join(concept_set),]
                if 'ke' in self.experiment:
                    description = ''
                    if 'dpr' in self.experiment:
                        description_dict = data['description_dict']
                        for concept in concept_set:
                            temp_list = description_dict[concept] if description_dict[concept] != 'MISMATCH' else ' '
                            temp = temp_list[0]
                            temp = temp + '.' if temp[-1] != '.' else temp
                            description += f'{concept.lower()}: {temp.lower()}; '
                    else:
                        for concept in concept_set:
                            wkdt = self.wkdt.match_description(concept)
                            if wkdt['matched'] != "MISMATCH":
                                des = f"{concept}: {wkdt['desc_list'][0]}"
                                des = des + '.' if des[-1] != '.' else des
                                description += des.lower() + '; '
                                
                    if 'proto' in self.experiment:
                        description += '' + data['prototype'][0]
                    example.insert(0, description)
                else:
                    if 'proto' in self.experiment:
                        example.insert(0, data['prototype'][0])
                
                example_id = f'{self.dataset_type}.{index}'
                self.raw_examples.append([example_id, example])
        logging.info(f"{self.dataset_type} dataset loaded")

    class Convert:
        # tokenize and get the max len   for multiprocessing
        def __init__(self, 
            tokenizer, 
            evaluate,
            max_src_len=50, max_tgt_len=50,
            experiment = ''
            ):
            self.tokenizer = tokenizer
            self.evaluate = evaluate
            self.max_src_len = max_src_len   # for bart encoder
            self.max_tgt_len = max_tgt_len   # for gpt2 & bart decoder
            self.experiment = experiment

        def __call__(self, raw_example):
            try:
                return self._bart(raw_example)
            except:
                logging.warning(raw_example)
                exit()

        def _bart(self, raw_example):
            scene = raw_example[-1]
            if 'ke' in self.experiment or 'proto' in self.experiment:
                description, concept = raw_example[:2]
                source_inputs = self.tokenizer(description, concept, 
                    max_length=self.max_src_len, padding='max_length', truncation='longest_first')
            else:
                concept = raw_example[0]
                source_inputs = self.tokenizer(concept, 
                    max_length=self.max_src_len, padding='max_length', truncation=True)
                
            target_inputs = self.tokenizer.encode(scene,
                max_length=self.max_tgt_len, padding='max_length', truncation=True)

            target_inputs.insert(0, self.tokenizer.sep_token_id)
            feature_dict = source_inputs
            feature_dict['decoder_input_ids'] = target_inputs
            return feature_dict
    
    def convert_tokens_to_ids(self):
        '''
        make input and label
        tokenized_examples: 
        -gpt2 [question, answer] 
        -bart [
            source: <CLS> answer <SEP> passage <SEP>
            target: <SEP> <CLS> question <SEP>
            ]
        '''
        
        logging.info(f"tokenizing {self.dataset_type} examples")
        logging.info(f'data format {self.raw_examples[0]}')
        
        now = time.time()
        # import pdb; pdb.set_trace()
        with Pool(processes=min(8, cpu_count())) as pool:
            tokenized_examples = pool.map(
                self.Convert(self.tokenizer, self.evaluate, self.max_src_len, self.max_tgt_len, self.experiment),
                self.raw_examples)
        logging.info(f"start {min(8, cpu_count())} processes, cost {time.time() - now}")
        

        for feature_dict in tokenized_examples:
            label = feature_dict['decoder_input_ids'][1:]
            decoder_input_ids = feature_dict['decoder_input_ids'][:-1]# no need to input the last token 
            feature_dict['decoder_input_ids'] = decoder_input_ids
            self.examples.append(feature_dict)

            for index, token in enumerate(label):
                if token == self.tokenizer.pad_token_id:
                    label[index] = -100
            self.labels.append(label)

    def make_dataloader(self, batch_size):
        if self.dataset_type == "train":
            data_sampler = RandomSampler(self)
        else:
            data_sampler = SequentialSampler(self)

        dataloader = DataLoader(self, sampler=data_sampler, batch_size=batch_size, num_workers=4, collate_fn=self.collate_fn) # TODO

        return dataloader

    def collate_fn(self, batch):

        pad_token_id = self.tokenizer.pad_token_id
        max_inputids_len = 0
        max_dec_inputids_len = 0
        for example in batch:
            input_ids = example['input_ids']
            decoder_input_ids = example['decoder_input_ids']
            for index, token in enumerate(input_ids):
                if token == pad_token_id:
                    max_inputids_len = max(max_inputids_len, index)
                    break
            else:
                max_inputids_len = len(input_ids)
            for index, token in enumerate(decoder_input_ids):
                if token == pad_token_id:
                    max_dec_inputids_len = max(max_dec_inputids_len, index)
                    break
        input_ids = []
        masks = []
        target_ids = []
        labels = []
        for example in batch:
            input_ids.append(example['input_ids'][:max_inputids_len])
            masks.append(example['attention_mask'][:max_inputids_len])
            target_ids.append(example["decoder_input_ids"][:max_dec_inputids_len])
            labels.append(example['labels'][:max_dec_inputids_len])
        
        input_ids = torch.stack(input_ids)
        masks = torch.stack(masks)
        target_ids = torch.stack(target_ids)
        labels = torch.stack(labels)
                
        batch = {
            "input_ids": input_ids,
            "attention_mask": masks,
            "decoder_input_ids": target_ids,
            "labels": labels
        }
        return batch

    def __len__(self):
        if self.evaluate:
            return len(self.raw_examples)
        else:
            return len(self.examples)

    def __getitem__(self, idx):
        if self.evaluate:
            return self.raw_examples[idx]
            
        else:
            feature_dict= {}
            for key, value in self.examples[idx].items():
                feature_dict[key] = torch.tensor(value)
            feature_dict['labels'] = torch.tensor(self.labels[idx])
            return feature_dict


if __name__ == "__main__":
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/bart-large")
    
    target_type = 'commongen'
    
    if target_type == 'protoqa':
        # ProtoQA test
        dataset_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1"
        
        dataset = ProtoQA_Dataset(
            dataset_path, model_type='bart',
            max_src_len=40, max_tgt_len=45, 
            overwrite_cache=True, tokenizer=tokenizer, 
            dataset_type='train', evaluate=False,
            wkdt_path='', experiment='',
        )
    elif target_type == 'commongen':
        dataset_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/CommonGen/description_prototype_v2"
        dataset = CommonGen_Dataset(
            dataset_path, max_src_len=10, 
            max_tgt_len=50, tokenizer=tokenizer,
            # evaluate=True,
            dataset_type='dev',
            experiment='ke_proto'
            
            # experiment='ke_dpr'
        )
        
    batch = [dataset[0],dataset[33]]
    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # dataset.collate_fn(batch)
    dataloader = dataset.make_dataloader(4)
    # for batch in dataloader:
        # import pdb; pdb.set_trace()
    