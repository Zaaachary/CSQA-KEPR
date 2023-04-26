from nltk.corpus import wordnet as wn
# from nltk.corpus import framenet as fn


def word_synset(word):  # 通过一个关键词，获得该词在Wordnet和framenet的所有同义词
    """
    @parameter : word 关键词
    @return : word_synset 同义词集（或列表）
    """
    word_synset = set()

    # TODO 获取WordNet中的同义词集
    synsets = wn.synsets(word)  # word所在的词集列表
    for synset in synsets:
        words = synset.lemma_names()
        for word in words:
            word = word.replace('_', ' ')
            word_synset.add(word)
#     print(list(word_synset))

    # TODO 获取FrameNet中的词组等
    # pattern = r'(?i)%s' % word  # 正则匹配表达式
    # frames = fn.frames(pattern) # 获取FrameNet中所有相关框架
    # for frame in frames:
    #     words = [x for x in frame.FE]
    #     words += words
    # for word in words:
    #     word_synset.add(word)
#     print(list(word_synset))

    word_synset = list(word_synset)
    return word_synset


if __name__ == "__main__":
    print(word_synset('shade'))