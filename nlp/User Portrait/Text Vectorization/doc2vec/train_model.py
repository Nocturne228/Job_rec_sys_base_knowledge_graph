#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim.models as g
import jieba
from gensim.corpora import WikiCorpus
import logging
from langconv import *
import pickle

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docvec_size = 192


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        import jieba
        for content, (page_id, title) in self.wiki.get_texts():
            yield g.doc2vec.TaggedDocument(
                words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])


def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
    """
    对给定的文本进行分词处理，包括繁简体转换和结巴分词。

    Parameters:
    - text: str, 待处理的文本字符串。
    - token_min_len: int, 生成的分词的最小长度。
    - token_max_len: int, 生成的分词的最大长度。
    - lower: bool, 是否将文本转换为小写。

    Returns:
    - list: 处理后的分词列表。
    """
    # 首先进行繁简体转换
    text = Converter('zh-hans').convert(text)
    if lower:
        text = text.lower()
    # 使用结巴分词进行分词
    tokens = list(jieba.cut(text))
    # 过滤分词长度和空白分词
    tokens = [token for token in tokens if token_min_len <= len(token) <= token_max_len and token.strip()]
    return tokens


def train():
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name, tokenizer_func=tokenizer_func)
    documents = TaggedWikiDocument(wiki)

    # 保存 documents 对象到文件
    with open('documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    model = g.Doc2Vec(documents, dm=0, dbow_words=1, vector_size=docvec_size, window=8, min_count=19, epochs=5, workers=8)
    model.save('data/zhiwiki_news.doc2vec')


if __name__ == '__main__':
    train()
