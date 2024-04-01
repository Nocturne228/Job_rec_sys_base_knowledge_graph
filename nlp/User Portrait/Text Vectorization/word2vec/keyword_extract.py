# -*- coding: utf-8 -*-
import jieba.posseg as pseg
from jieba import analyse

"""
从文本文件中提取关键词并保存到文件中
"""


def keyword_extract(data, file_name):
    # 对文本数据进行关键词提取，并将提取出的关键词返回
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords


def getKeywords(docpath, savepath):
    """
    :param docpath: 待提取关键词的文本文件路径
    :param savepath: 保存提取出的关键词的文件路径
    :return: 将提取出的关键词保存到文件中
    """
    with open(docpath, 'r', encoding='utf-8') as docf, open(savepath, 'w', encoding='utf-8') as outf:
        for data in docf:
            data = data[:len(data) - 1]
            keywords = keyword_extract(data, savepath)
            for word in keywords:
                outf.write(word + ' ')
            outf.write('\n')
