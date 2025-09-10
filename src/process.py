import pandas as pd
from tqdm import tqdm

import config
from sklearn.model_selection import train_test_split
import jieba

'''
预处理词向量
'''


def build_dataset(sentences, word2index):
    '''
    构建数据集
    :param sentences: 原始句子列被['我爱nlp,我不爱nlp]
    :param word2index: 词向量表
    :return: [{input:[1,2,3,4,5],target:6},{input:[2,3,4,5，6],target:7}.....]
    '''
    # 如果get为none就赋值为0索引,获取句子所有词汇索引
    indexed_sentences = [[word2index.get(word, 0) for word in jieba.lcut(sentence)] for sentence in sentences]
    # [{input:[1,2,3,4,5],target:6},{input:[,2,3,4,5，6],target:7}.....]
    dataset = []
    for sentence in indexed_sentences:
        # sentence :[1,2,3,4,5]
        for i in range(len(sentence) - config.SEQ_LEN):
            #切片滑动窗口
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset


def process():
    '''
    预处理数据
    :return:
    '''
    print('开始处理')
    # 1、读取数据
    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl', orient='records', lines=True)
    print(df.head())

    # 2.抽取处理数据dialog
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            text = sentence.split('：')[1]
            sentences.append(text)
    print(f'第一句：{sentences[0]}')
    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=config.RANDOM_SEED)

    # 构建词表，训练集
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc='构建词表'):
        for word in jieba.lcut(sentence):
            vocab_set.add(word)
    vocab_list = ['<unk>'] + list(vocab_set)
    print(f'词表的大小:{len(vocab_list)}')
    # 保存词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(word + "\n")
    print(f'词表保存完成:{len(vocab_list)}')

    # 构建map词表,从去重后的集合中。去重、切词、构建都是使用jieba分词
    word2index = {word: index for index, word in enumerate(vocab_list)}
    # 构建并保存训练集
    # 分词，索引，滑动窗口

    train_dataset = build_dataset(train_sentences, word2index)
    test_dataset = build_dataset(test_sentences, word2index)

    # 保存训练数据集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DIR / 'index_train.jsonl', orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DIR / 'index_test.jsonl', orient='records', lines=True)

    print()

    # 构建并报存测试集

    print('结束处理')


if __name__ == '__main__':
    process()
