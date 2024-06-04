from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
import numpy as np


class BPEEncoder:
    def __init__(self, vocab_path, subword_map_path):
        with codecs.open(vocab_path, 'r', 'utf-8') as file:
            self.bpe = BPE(file, merges=-1, separator='')

        sub_csv = pd.read_csv(subword_map_path)
        idx2word = sub_csv['index'].values
        self.words2idx = dict(zip(idx2word, range(0, len(idx2word))))

    def encode(self, x, max_len=50, mask = True):
        tokens = self.bpe.process_line(x).split()

        try:
            indices = np.asarray([self.words2idx[token] for token in tokens])
        except KeyError:
            indices = np.array([0])

        if(mask):
            l = len(indices)
            if l < max_len:
                padded_indices = np.pad(indices, (0, max_len - l), 'constant', constant_values=0)
                input_mask = ([1] * l) + ([0] * (max_len - l))
            else:
                padded_indices = indices[:max_len]
                input_mask = [1] * max_len

            return padded_indices, np.asarray(input_mask)
        else:
            v1 = np.zeros(len(self.words2idx),)
            v1[indices] = 1
            return v1	


