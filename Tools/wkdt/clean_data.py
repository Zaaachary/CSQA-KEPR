'''
pos: adj, adv, noun, name, verb
'''
import json
import re

from tqdm import tqdm


data_dir = '/data1/zhifli/WKDT/kaikki.org-dictionary-English.json'
ouput_data = '/data1/zhifli/WKDT/dictionary-English-clean.format.json'
ouput_data2 = '/data1/zhifli/WKDT/dictionary-English-clean.json'

del_list1 = [
    "translations", "sounds", "derived", "hyponyms", "related", "hyphenation", "heads", "lang", "categories", "wikipedia", "forms", "lang_code"
    ]
del_list2 = [
    "translations", "derived", "hypernyms", "hyponyms", "categories", "tags", "related", "form_of", "id", "wikipedia"
    ]

word_rm_pattern = re.compile(r"[^a-zA-Z0-9.\-]")

def clean_word(word_case):
    # 去掉非字母 非数字 非 . - 词
    if re.match(word_rm_pattern, word_case['word']):
        return None
    elif word_case['word'][0] == '-':
        return None

    for del_case in del_list1:
        if del_case in word_case:
            del word_case[del_case]

    senses = word_case['senses']
    new_senses = []
    for sense in senses[::]:
        if 'glosses' in sense:
            # 列表转字符串
            sense['glosses'] = sense['glosses'][0]
        else:
            continue

        for del_case in del_list2:
            if del_case in sense:
                del sense[del_case]
        new_senses.append(sense)
        
    word_case['senses'] = new_senses

    return word_case

def sort_word(word_list):
    word_list.sort(key=lambda x:x['word'])

if __name__ == "__main__":
    f = open(data_dir, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    word_list = []
    for line in tqdm(lines):
        word = json.loads(line)
        word = clean_word(word)
        if word:
            word_list.append(word)

    sort_word(word_list)

    f = open(ouput_data, 'w', encoding='utf-8')
    json.dump(word_list, f, ensure_ascii=False, indent=2)
    f.close

    f = open(ouput_data2, 'w', encoding='utf-8')
    for word in word_list:
        jstr = json.dumps(word, ensure_ascii=False)
        f.write(jstr + '\n')
    f.close
