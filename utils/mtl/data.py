import numpy as np

import torch
from torch.utils.data import Dataset


class CreatorDataset(Dataset):
    '''
    Класс для создания структуры датасета, используемого при обучении НН
    '''

    def __init__(self, corpus, targets, maxlen=30):

        '''
        corpus - список последовательностей
        targets - метки класса
        maxlen - максимальная длина последовательностиы
        '''

        self.padded_corpus = []
        self.targets = targets
        self.len = len(corpus)

        for seq in corpus:
            seqlen = len(seq)
            if seqlen <= maxlen:
                obj = [0] * (maxlen - seqlen) + seq
            else:
                obj = seq[-maxlen:]

            self.padded_corpus.append(obj)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.tensor(self.padded_corpus[idx]), torch.tensor(self.targets[idx])
