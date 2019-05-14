# PATHS
# training : 'project_2_data/02-21.10way.clean'
# validation: 'project_2_data/22.auto.clean'
# test: 'project_2_data/23.auto.clean'
from collections import defaultdict
import numpy as np
import torch


class Data_reader:

    def __init__(self, path=None):

        self.path = path
        self.w2i = defaultdict(int)
        self.i2w = ['UNK']
        self.w2i['UNK'] = 0
        self.sentences, sentences_indices = self.preprocess_data(path)
        self.sentences_indices = np.array(sentences_indices)
        np.random.shuffle(self.sentences_indices)

    def preprocess_to_sent(self, line):

        sent = []
        sent_indices = []
        for i, sym in enumerate(line):
            if sym == ')' and line[i - 1] != ')' and line[i - 1] != ' ':
                word_start = 0
                c = None
                while c != ' ':

                    c = line[i - word_start]
                    word_start += 1

                word = line[i - word_start + 2:i]

                sent.append(word)

                if word not in list(self.w2i.keys()):
                    self.w2i[word] = len(self.w2i.keys())
                    self.i2w.append(word)

                sent_indices.append(self.w2i[word])
        return sent, sent_indices

    def preprocess_data(self, path):
        corpus = []
        corpus_indices = []
        with open(path) as f:
            for line in f:
                sent, sent_indices = self.preprocess_to_sent(line)
                if len(sent_indices) == 0:
                    continue
                corpus.append(sent)
                corpus_indices.append(sent_indices)

        return corpus, corpus_indices

    def create_batches(self, batch_size):
        self.sentences_indices = self.sentences_indices.tolist()
        for i in range(0, len(self.sentences_indices), batch_size):
            batch = self.sentences_indices[i:i + batch_size]
            batch.sort(key=len, reverse=True)
            batch = [torch.tensor(s) for s in batch]
            yield batch
