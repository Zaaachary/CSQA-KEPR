# -*- encoding: utf-8 -*-
'''
@File    :   match.py
@Time    :   2022/02/28 23:41:27
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
from ast import dump
import os
import sys
import pickle
import logging
import re
from collections import OrderedDict

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from numpy import isin
import inflect
logging.basicConfig(level=logging.INFO)  


class WKDT_matcher:

    _rm_pattern = re.compile(r'[\?\!\.]')
    _trans_dict = {'n':'noun', 'v':'verb', 'a':'adj', 'r':'adv'}

    def __init__(self, wkdt_path, experiment='', mode='top1_noun') -> None:
        self.wkdt = self.pickle_load(wkdt_path)
        self.lemmatizer = WordNetLemmatizer()
        self.mode = mode
        self.experiment = experiment

        self.infeg = inflect.engine()

    def match_description(self, target):
        '''
        strip
        - digit word -> return None
        - single word 
            -> remove non-english character -> lemmatisation
            -> origin single word
            -> remove non-english character
        - phrase -> left to right matching
        '''
        target = target.strip()
        word_list = target.split(' ')
        if target.isdigit():
            words = self.infeg.number_to_words(int(target))
            return {'matched': target, 'desc_list': [words, ], 'pos': 'num'}
        elif len(word_list) == 1:
            if 'pos' in self.experiment:
                describe_dict = self.single_word_pos(target)
            else:
                describe_dict = self.single_word(target)
            
        elif len(word_list) >= 2:
            describe_dict = self.phrase(word_list)

        if 'MISMATCH' not in describe_dict:
            if self.mode == "top1_noun":
                if "pos" in self.experiment and len(word_list) == 1:
                    return describe_dict
                matched_term, pos, desc_list = self.select_describe(describe_dict)
                return {'matched': matched_term, 'desc_list': desc_list, 'pos': pos}
            elif self.mode == 'list':
                matched_term, desc_list = self.merge2list(describe_dict)
                return {'matched': matched_term, 'desc_list': desc_list}
        else:
            return {'matched': "MISMATCH"}

    def single_word_pos(self, target):
        target_en = re.sub(self._rm_pattern, '', target)
        tag = pos_tag([target_en, ])[0]
        wordnet_pos = self.get_wordnet_pos(tag[1])
        if wordnet_pos:
            target_en_lemma = self.lemmatizer.lemmatize(tag[0], pos=wordnet_pos)
        else:
            target_en_lemma = self.lemmatizer.lemmatize(tag[0])

        total_result = OrderedDict()
        for tg in [target_en_lemma, target_en]:
            if tg not in total_result:
                result = self.wkdt.get(tg, None)
                if result:
                    total_result[tg] = result
                    
        if total_result:
            wkdt_pos = self._trans_dict.get(wordnet_pos, None)

            matched_word, description_dict = total_result.popitem(0)
            if wkdt_pos and wkdt_pos in description_dict:
                return {'matched': matched_word, 'desc_list': description_dict[wkdt_pos][:3], 'pos':wkdt_pos}
            elif self.mode == 'list':
                temp = {}
                for pos, desc_list in description_dict.items():
                    temp[pos] = desc_list
                return {target: temp}
            
            elif 'noun' in description_dict:
                return {'matched': matched_word, 'desc_list': description_dict['noun'][:3], 'pos':'noun'}
            else:
                pos, desc_list = description_dict.popitem()
                assert isinstance(desc_list, list)
                return {'matched': matched_word, 'desc_list': desc_list[:3], 'pos':pos}
        else:
            return {'MISMATCH': ''}

    def single_word(self, target):
        target_en = re.sub(self._rm_pattern, '', target)
        target_lemma = self.lemmatizer.lemmatize(target)
        target_en_lemma = self.lemmatizer.lemmatize(target_en)

        total_result = OrderedDict()
        for tg in [target, target_en, target_lemma, target_en_lemma]:
            if tg not in total_result:
                result = self.wkdt.get(tg, None)
                if result:
                    total_result[tg] = result
                    
        if total_result:
            return total_result
        else:
            return {'MISMATCH': ''}

    def phrase(self, word_list):
        no_space = [''.join(word_list), ]

        word_list_en = [re.sub(self._rm_pattern, '', word)
                        for word in word_list]
        word_list_lemma = [self.lemmatizer.lemmatize(
            word) for word in word_list]
        word_list_en_lemma = [self.lemmatizer.lemmatize(
            word) for word in word_list_en]

        total_result = OrderedDict()
        for wl in [no_space, word_list, word_list_en, word_list_lemma, word_list_en_lemma]:
            for start in range(len(wl)):
                target = " ".join(wl[start:])
                if target not in total_result:
                    result = self.wkdt.get(target, None)
                    if result:
                        total_result[target] = result

        for wl in [word_list, word_list_en, word_list_lemma, word_list_en_lemma]:
            for end in range(len(wl)-1, -1, -1):
                target = " ".join(wl[:end])
                if target not in total_result:
                    result = self.wkdt.get(target, None)
                    if result:
                        total_result[target] = result

        if total_result:
            return total_result
        else:
            return {'MISMATCH': ''}

    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def select_describe(describe_dict, mode='noun'):
        for key, value in describe_dict.items():
            pos, desc_list = '', []
            if 'noun' in value:
                pos = 'noun'
                desc_list = value['noun'][:3]
            elif len(value) > 0:
                pos, desc_list = value.popitem()
            
            return key, pos, desc_list

    @staticmethod
    def merge2list(describe_dict):
        total_dict = {}
        key_str = ''
        for key, value in describe_dict.items():
            key_str += f"{key};"
            for pos, desc in value.items():
                temp = total_dict.get(pos, [])
                temp.extend(desc[:3])
                total_dict[pos] = list(set(temp))
            
        return key_str, total_dict

    @staticmethod
    def pickle_load(path):
        f = open(path, 'rb')
        wiktionary = pickle.load(f)
        f.close()
        return wiktionary

def tarverse_and_format(data, wkdt):
    for example in data:
        # question = example['question']['normalized']
        keyword_list = example['keyword']
        example['keyword_description'] = dict.fromkeys(keyword_list, [])
        for keyword in keyword_list:
            description_info = wkdt.match_description(keyword)

            if description_info['matched'] == 'MISMATCH':
                description_info['desc_list'] = []
            else:
                desc_list = []
                for key, value in description_info['desc_list'].items():
                    desc_list.extend(value)
                description_info['desc_list'] = desc_list
            
            example['keyword_description'][keyword] = description_info['desc_list']


    return data


if __name__ == "__main__":
    # pass
    # sys.path.append('../')
    # from ..data_io_util import load_data, dump_data
    # wkdt = WKDT_matcher("/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.pkl", mode='list')

    # input_root = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword"
    # output_root = "/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/description_data/v1"

    # data_list = ['dev.crowdsourced.jsonl', 'dev.scraped.jsonl', 'test.questions.jsonl', 'train.scraped.jsonl']
    # input_data_path = [os.path.join(input_root, data) for data in data_list]
    # output_data_path = [os.path.join(output_root, data) for data in data_list]

    # for input_path, output_path in zip(input_data_path, output_data_path):
    #     data = load_data(input_path, 'jsonl')
    #     data = tarverse_and_format(data, wkdt)
    #     dump_data(data, output_path, mode='json')

    # target = ['guess', 'called', 'monk', 'leave', 'sugar', 'head', 'luxury', 'equipment', 'garage', 'safe', 'lasts', 'government']
    target = ['leave']
    # target = ['millions', 'mispronounce', 'require']
    # target = ['lasts']
    wkdt = WKDT_matcher("/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict", experiment='pos', mode='list')
    # wkdt = WKDT_matcher("/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict", experiment='pos', mode='top1_noun')
    # wkdt = WKDT_matcher("/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.pkl", experiment='pos', mode='top1_noun')
    for word in target:
        # wkdt.match_description(word)
        print(word, wkdt.match_description(word))
