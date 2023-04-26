#! -*- encoding:utf-8 -*-
"""
@File    :   extract_sense.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   

word:{
    'noun' : []
    'verb' : []
}
"""
import json
import pickle
from pprint import pprint

from tqdm import tqdm

data_dir = r"/data1/zhifli/WKDT/dictionary-English-clean.json"
output_path = r"/data1/zhifli/WKDT/wiktionary.pkl"
output_path2 = r"/data1/zhifli/WKDT/wiktionary.json"


def load_wiktionary(data_dir):
    f = open(data_dir, 'r', encoding='utf-8')
    lines = []
    for line in f.readlines():
        lines.append(line.strip()) 
    f.close()
    return lines

def clean_sense(case):
    # 只保留 sense 中的 glosses
    place = len(case['senses']) - 1
    for sense in case['senses'][::-1]:
        glosses = sense.get('glosses', None)
        sense.clear()
        if glosses is None:
            del case['senses'][place]
        else:
            sense['glosses'] = glosses
        place -= 1

def traverse(lines):
    word_list = []
    for case in tqdm(lines):
        case_dict = json.loads(case)
        senses = case_dict['senses']
        pos = case_dict['pos']
        word = case_dict['word']
        case_dict = {'word':word, 'senses':senses, 'pos': pos}
        clean_sense(case_dict)
        word_list.append(case_dict)
    return word_list

def make_dictionary(word_list):
    wiktionary = {}
    for case in tqdm(word_list):
        # word = case['word'].lower()
        word = case['word']

        if not wiktionary.get(word, None):
            wiktionary[word] = {}       # {noun:[], verb:[]}

        for glosse in case['senses']:
            pos = case['pos']
            wiktionary[word][pos] = wiktionary[word].get(pos, [])
            wiktionary[word][pos].append(glosse['glosses'])
        # wiktionary[word] = list(set(wiktionary[word]))
    return wiktionary

def rdf_plural(wiktionary):
    for word, value in wiktionary.items():
        noun = value.get('noun', None)
        if noun and 'plural of ' in noun[0]:
            origin = noun[0][10:].replace('.','')
            origin = wiktionary.get(origin, None)
            # origin = wiktionary.get(origin.lower(), None)
            if origin:
                wiktionary[word] = origin

def pickle_dump(output_path, obj):
    f = open(output_path, 'wb')
    pickle.dump(obj, f)
    f.close()

def pickle_load(path):
    f = open(path, 'rb')
    wiktionary = pickle.load(f)
    f.close()
    return wiktionary

def save_data(data_dir, data):
    f = open(data_dir, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.close()

def main():
    lines = load_wiktionary(data_dir)
    word_list = traverse(lines)
    wiktionary = make_dictionary(word_list)
    rdf_plural(wiktionary)

    pickle_dump(output_path, wiktionary)
    save_data(output_path2, wiktionary)

def test():
    wiktionary = pickle_load(output_path)

    try:
        while True:
            key = input('input a word:')
            print(wiktionary.get(key, 'not here'))
    except EOFError:
        pprint(wiktionary[:30])

if __name__ == "__main__":
    main()
    # test()