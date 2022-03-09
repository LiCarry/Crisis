import gensim
import math
import jieba
import jieba.posseg as posseg
from jieba import analyse
from gensim import corpora, models
import functools
import numpy as np
import os
import time
from tqdm import tqdm


# 加载整个文件下的数据
def load_whole_dataSet(datafolder_path):
    prepared_data = []
    files = os.listdir(datafolder_path)
    for file in files:
        if not os.path.isdir(datafolder_path + file):  # 判断是否是文件夹，不是文件夹才打开
            for line in open(datafolder_path + "/" + file, 'r', encoding='utf-8'):
                prepared_data.append(line)
    return prepared_data


# 对加载的数据预处理
def pre_dataSet(prepared_data, pos=False):
    doc_list = []
    for line in prepared_data:
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filetr_list = word_filter(seg_list, pos)
        doc_list.append(filetr_list)
    return doc_list


# 停用词
def get_stopword_list(stopword_path):
    stopword_list = [stopword.replace('\n', ' ') for stopword in open(stopword_path, encoding='gbk').readlines()]
    return stopword_list


# jieba分词
def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = posseg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, stopword_path, pos=False):
    stopword_list = get_stopword_list(stopword_path)
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if word not in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list


# 排序函数， 用于topK关键字按值排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.tfidf_corpus = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载胡模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    # 向量化
    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        print("vec_list", vec_list)
        return vec_list

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        # 2级变1级结构
        for doc in doc_list:
            # extend和append 方法有何异同 容易出错
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    # 得到数据集的主题 - 词分布
    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            singlist = [word]
            # 计算每个词的加权向量
            word_corpus = self.tfidf_model[self.dictionary.doc2bow(singlist)]
            # 计算每个词的主题向量
            word_topic = self.model[word_corpus]
            wordtopic_dic[word] = word_topic

        return wordtopic_dic

    def train_lsi(self):
        lsi = models.LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        # 文档的加权向量
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        # 文档主题 向量
        senttopic = self.model[sentcorpus]

        # senttopic [(0, 0.03457821), (1, 0.034260772), (2, 0.8970413), (3, 0.034119748)]
        # 余弦相似度计算

        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            # 还是计算每个文档中的词和文档的相识度
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        counts = {}
        keyWordDict = []
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            if k is not None:
                keyWordDict.append(k)
        return keyWordDict


# 提取文件夹下所有文件的关键词，仅用到构建关键词字典
def load_whole_dataSet(datafolder_path, pos=False, model='LDA', keyword_num=100):
    prepared_data = []
    files = os.listdir(datafolder_path)
    counts = {}
    for file in tqdm(files):
        prepared_data = []
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            for line in open(datafolder_path + "/" + file, 'r', encoding='utf-8'):
                content = line.strip()
                seg_list = seg_to_list(content, pos)
                filetr_list = word_filter(seg_list, pos)
                prepared_data.append(filetr_list)
        if prepared_data:
            try:
                topic_model = TopicModel(prepared_data, keyword_num, model=model)
            except ValueError as e:
                pass
        for doc_list_i in prepared_data:
            keyWordDict = topic_model.get_simword(doc_list_i)
            for word in keyWordDict:
                if len(word) == 1:
                    continue
                else:
                    counts[word] = counts.get(word, 0) + 1
    return counts


if __name__ == '__main__':
    # 设置停用词路径
    stopword_path = '~/dataSet/stop_words/stop_words.txt'
    # ket_word_num 的数量大小在下一篇文章中给出
    keyword_num = 10
    # text为需要提取关键词的文本
    text = "^ ^"
    prepared_data = []
    content = text.strip()
    seg_list = seg_to_list(content, pos=False)
    filetr_list = word_filter(seg_list, pos=False)
    prepared_data.append(filetr_list)
    topic_model = TopicModel(prepared_data, keyword_num, model='LDA')
    counts = {}
    for item in prepared_data:
        keyWordDict = topic_model.get_simword(item)
        for word in keyWordDict:
            if len(word) == 1:
                continue
            else:
                counts[word] = counts.get(word, 0) + 1
    print(counts)
