import numpy as np
import pandas as pd
import gensim.downloader as api
import config


class GoogleW2V:
    def __init__(self):
        self.google_w2v = api.load('word2vec-google-news-300')

    def convert_to_vec(self, text):
        words = text.split()
        word_vec = []
        for word in words:
            try:
                w_vec = self.google_w2v[word]
                word_vec.append(w_vec)
            except KeyError:
                pass
        return np.array(word_vec)


class GloveW2V:
    def __init__(self, dim=300):
        assert dim in [50, 100, 200, 300]
        glove_path = eval(f"config.GLOVE_{dim}")
        df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        self.glove = {key: val.values for key, val in df.T.items()}
        del df

    def convert_to_vec(self, text):
        words = text.split()
        word_vec = []
        for word in words:
            try:
                w_vec = self.glove[word]
                word_vec.append(w_vec)
            except KeyError:
                pass
        return np.array(word_vec)

