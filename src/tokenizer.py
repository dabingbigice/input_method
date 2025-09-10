import jieba
from tqdm import tqdm

import config


class Tokenizer:
    unk_token = '<unk>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text):
        '''
        获取分词对应索引
        :param text:
        :return:
        '''
        word_list = Tokenizer.tokenize(text)
        return [self.word2index.get(word, self.unk_token_id) for word in word_list]

    @classmethod
    def from_vocab(cls, vocab_file):
        # 获取词表
        with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
            vocal_list = [line.strip() for line in f.readlines()]
        print('词表加载成功')
        return cls(vocal_list)

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        # 构建词表
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            for word in jieba.lcut(sentence):
                vocab_set.add(word)
        vocab_list = [cls.unk_token] + list(vocab_set)
        print(f'词表的大小:{len(vocab_list)}')
        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + "\n")
        print(f'词表保存完成:{len(vocab_list)}')


if __name__ == '__main__':
    tokenize = Tokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')
    print(tokenize.tokenize('我喜欢坐地铁'))
    print(tokenize.encode('我喜欢坐地铁'))
