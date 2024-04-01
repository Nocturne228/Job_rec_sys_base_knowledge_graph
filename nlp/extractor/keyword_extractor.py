import pandas as pd
import re


def add_keyword_to_df(csvfile, attr_name, encoding='utf-8'):
    """
    使用pandas读取csv文件，对已经分词过的attr_name列进行清洗；
    包括切分符号、去标点、去单字等等；
    输出一个dataframe格式的，标题为：'id', 'before', 'after' 的清洗后数据
    :param csvfile:csv文件路径
    :param attr_name:要进行清洗的属性列的名称
    :param encoding: csv编码格式，默认utf-8
    :return:dataframe格式的，标题为：'id', 'before', 'after' 的清洗后数据
    """
    df = pd.read_csv(csvfile, encoding=encoding)
    # 待清洗的数据命名为pre_data
    pre_data = pd.DataFrame(df[attr_name])

    # 加载停用词
    stop_words = open('hit_stopwords.txt', 'r', encoding='utf-8').readlines()
    stop_words = [w.strip() for w in stop_words]

    def data_preprocess(description_words):
        # 将所有其他标点符号转为,
        description_words = re.sub(r'[^\w\s]', ',', description_words)
        # 这里经过了一个str->list的过程
        # 使用,分词
        description_words = description_words.split(',')

        # 去停用词
        description_words = [w for w in description_words if w not in stop_words]

        # 去空格
        description_words = [w.strip() for w in description_words]

        # 将所有英文字母转为小写
        description_words = [w.lower() for w in description_words]

        # 去除字数小于2的值
        description_words = [w for w in description_words if len(w) >= 2]

        # 去空行
        description_words = [w for w in description_words if w]

        return description_words

    for index, token in pre_data.iterrows():
        # if index <= 7:
        text = token[attr_name]
        words = data_preprocess(text)
        pre_data.at[index, attr_name] = ",".join(words)

    print(pre_data)
    print(pre_data.head())
    print(pre_data.shape)
    pre_data.columns = ['keyword']

    print(pre_data.columns)

    return pre_data


if __name__ == '__main__':
    new_data = add_keyword_to_df('../dataset/process/tokens_after_hanlp.csv', '0')
    new_data.to_csv('output.csv')
