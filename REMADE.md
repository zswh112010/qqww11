
# 文本特征提取项目

## 代码核心功能说明

本项目实现了一个参数化特征提取机制，支持两种特征提取方式：
1. **高频词特征**：基于词频统计的特征选择方法
2. **TF-IDF加权特征**：基于词频-逆文档频率的特征加权方法

核心类 `FeatureExtractor` 提供了灵活的特征提取方式切换，通过参数 `mode` 指定使用哪种特征提取方法。

# 高频词模式与TF-IDF模式的方法介绍

## 1. 高频词模式

### 方法介绍
高频词模式通过统计语料中词语的出现频率，选取出现次数最多的Top-N词语作为特征。具体实现步骤：

1. **分词处理**：对文本进行分词
2. **词频统计**：统计每个词语的出现次数
3. **排序筛选**：按词频降序排列，取前N个词语

### 计算公式
$$ TF(w_i) = \frac{count(w_i)}{\sum_{k=1}^n count(w_k)} $$

## 2. TF-IDF模式

### 方法介绍
TF-IDF（Term Frequency-Inverse Document Frequency）通过衡量词语在文档中的重要性进行特征提取，结合以下两个维度：

1. **词频（TF）**：词语在当前文档中的出现频率
2. **逆文档频率（IDF）**：词语在整个语料库中的分布稀有性

### 计算公式
$$ TF−IDF(t,d)=TF(t,d)×IDF(t) $$

# 文本特征提取方法：高频词与TF-IDF切换指南

## 核心方法对比
| 特性                | 高频词模式                     | TF-IDF模式                     |
|---------------------|------------------------------|-------------------------------|
| **计算维度**         | 单文档词频统计                | 跨文档词频-逆文档频率综合计算     |
| **特征权重**         | 原始词频/二值化              | TF × IDF 加权值               |
| **空间复杂度**       | O(V) V=词汇表大小            | O(V×D) D=文档数               |
| **适用场景**         | 实时分析/快速原型             | 语义分析/文档检索              |
| **典型工具**         | `CountVectorizer`           | `TfidfVectorizer`            |

# 示例代码
import re
import os
from jieba import cut
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmailClassifier:
    def __init__(self, mode='tfidf', top_num=100):
        """
        初始化邮件分类器

        参数:
        mode (str): 特征提取模式，"frequency" 或 "tfidf"
        top_num (int): 选择的特征数量
        """
        self.mode = mode
        self.top_num = top_num
        self.vectorizer = None
        self.model = None
        self.labels = None

    def get_words(self, filename):
        """读取文本并过滤无效字符和长度为1的词"""
        words = []
        try:
            with open(filename, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    # 过滤无效字符
                    line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                    # 使用jieba.cut()方法对文本切词处理
                    line = cut(line)
                    # 过滤长度为1的词
                    line = filter(lambda word: len(word) > 1, line)
                    words.extend(line)
        except FileNotFoundError:
            logging.error(f"文件 {filename} 不存在")
        except Exception as e:
            logging.error(f"处理文件 {filename} 时出错: {e}")
        return words

    def extract_features(self, corpus):
        """提取特征"""
        if self.mode == 'frequency':
            self.vectorizer = CountVectorizer(max_features=self.top_num)
        elif self.mode == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.top_num)
        else:
            raise ValueError("不支持的模式，仅支持 'frequency' 或 'tfidf'")

        X = self.vectorizer.fit_transform(corpus)
        return X

    def train(self, filename_list, labels):
        """训练模型"""
        corpus = []
        for filename in filename_list:
            words = self.get_words(filename)
            corpus.append(' '.join(words))

        X = self.extract_features(corpus)
        self.model = MultinomialNB()
        self.model.fit(X, labels)
        logging.info("模型训练完成")

    def predict(self, filename):
        """对未知邮件分类"""
        words = self.get_words(filename)
        text = ' '.join(words)
        try:
            current_vector = self.vectorizer.transform([text])
            result = self.model.predict(current_vector)
            return '垃圾邮件' if result == 1 else '普通邮件'
        except Exception as e:
            logging.error(f"预测文件 {filename} 时出错: {e}")
            return None


if __name__ == "__main__":
   
    mode = 'tfidf'  # 或 'frequency'
    top_num = 100
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    labels = np.array([1] * 127 + [0] * 24)

    # 初始化分类器
    classifier = EmailClassifier(mode=mode, top_num=top_num)

    # 训练模型
    classifier.train(filename_list, labels)

    # 预测未知邮件
    test_files = ['151.txt', '152.txt', '153.txt', '154.txt', '155.txt']
    for filename in test_files:
        result = classifier.predict(f'邮件_files/{filename}')
        print(f'{filename}分类情况: {result}')
        
# 示例代码结果图
<img width="890" alt="3 1" src="https://github.com/zswh112010/qqww11/blob/main/task4/classify%E7%BB%93%E6%9E%9C%E5%9B%BE.png" />

# classify结果图
<img width="890" alt="3 1" src="https://github.com/zswh112010/qqww11/blob/main/task4/%E7%A4%BA%E4%BE%8B%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%9C%E5%9B%BE.png" />


        
