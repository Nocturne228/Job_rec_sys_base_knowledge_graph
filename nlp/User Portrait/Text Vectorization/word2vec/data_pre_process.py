# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
import jieba
# 繁简体转换
from langconv import *


# def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
#     return [token for token in text.split() if token_min_len <= len(token) <= token_max_len]
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


def extrac_tok():
    """
    从维基百科的 XML 数据文件中提取中文文本，并进行繁简体转换和分词处理，然后将处理后的文本保存到文件中；
    从指定的维基百科 XML 数据文件中读取中文文本，并使用繁简体转换工具将文本转换为简体中文。
    然后，使用结巴分词工具对文本进行分词处理，并将分词结果保存到文件中。
    分词后的文本会被写入到名为 "reduce_zhiwiki.txt" 的文件中，该文件会在当前目录下生成或覆盖。
    在处理过程中，会在每处理完 200 篇文章时打印提示信息，以指示处理进度。
    :return: ./data/reduce_zhiwiki.txt
    """
    space = ' '
    i = 0
    l = []
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduce_zhiwiki.txt', 'w', encoding='utf-8')
    # FIXME
    # wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    wiki = WikiCorpus(zhwiki_name, tokenizer_func=tokenizer_func)
    for text in wiki.get_texts():
        for temp_sentence in text:
            temp_sentence = Converter('zh-hans').convert(temp_sentence)
            seg_list = list(jieba.cut(temp_sentence))
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if i % 200 == 0:
            print('Saved ' + str(i) + ' articles')
    f.close()


if __name__ == '__main__':
    extrac_tok()
