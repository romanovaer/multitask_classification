import numpy as np

import torch

from collections import Counter


class Vocabulary:

    '''
    Класс, кодирующий элементы последовательности
    '''

    def __init__(self, max_vocab_size, min_freq=1):

        '''
        max_vocab_size - максимальный размер словаря
        min_frec - минимальное количество раз, которое элемент должен
                   встретиться в последовательности для включения в словарь
        '''

        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab = None
        self.vocab_size = None
        self.vocab2id = None
        self.id2vocab = None

    def transform(self, corpus):

        if self.vocab is None:
            cnt = Counter()
            for doc in corpus:
                cnt.update(doc)
            self.vocab = [value for value, freq in
                          cnt.most_common(self.max_vocab_size)
                          if freq >= self.min_freq]
            self.vocab_size = len(self.vocab) + 1
            self.vocab2id = dict(zip(self.vocab, range(1, self.vocab_size)))
            self.vocab2id.update({'UNK': 0})
            self.id2vocab = ['UNK'] + self.vocab

        return [[self.vocab2id.get(word, 0) for word in doc] for doc in corpus]
