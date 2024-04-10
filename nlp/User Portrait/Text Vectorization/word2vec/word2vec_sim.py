# -*- coding: utf-8 -*-
import codecs
import numpy
import gensim
import numpy as np
from keyword_extract import *


wordvec_size = 100


def get_char_pos(string, char):
    chPos = []
    try:
        chPos = list(((pos) for pos, val in enumerate(string) if (val == char)))
    except:
        pass
    return chPos


def word2vec(file_name, model):
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        word_vec_all = numpy.zeros(wordvec_size)
        for data in f:
            space_pos = get_char_pos(data, ' ')
            first_word = data[0:space_pos[0]]
            # if model.__contains__(first_word):
            if first_word in model.wv:
                word_vec_all = word_vec_all + model.wv.get_vector(first_word)

            for i in range(len(space_pos) - 1):
                word = data[space_pos[i]:space_pos[i + 1]]
                # if model.__contains__(word):
                if word in model.wv:
                    word_vec_all = word_vec_all + model.wv.get_vector(word)
        return word_vec_all


def calc_similarity(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        similarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        similarity = 0
    return similarity


if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('zhiwiki_news.word2vec')
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywords.txt'
    p2_keywords = './data/P2_keywords.txt'
    getKeywords(p1, p1_keywords)
    getKeywords(p2, p2_keywords)
    p1_vec = word2vec(p1_keywords, model)
    p2_vec = word2vec(p2_keywords, model)
    print(p1_vec)
    print(p2_vec)

    print(calc_similarity(p1_vec, p2_vec))
