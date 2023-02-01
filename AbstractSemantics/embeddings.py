import nltk
import re
import numpy as np
import multiprocessing as mp

import itertools

from typing import List

from abc import ABC, abstractmethod


class Embeddings(ABC):
    def __init__(self, tokens: List[List[str]]):
        self.tokens = tokens

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def recover(self):
        pass


class GensimWord2Vec(Embeddings):
    def __init__(self, tokens, **kwargs):
        super().__init__(tokens)

    def train(
        self,
        vector_size: int = 128,
        window: int = 20,
        min_count: int = 10,
        threads: int = 4,
        **kwargs
    ):
        from gensim.models import word2vec

        model = word2vec.Word2Vec(
            self.tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=threads,
            **kwargs
        )
        return model

    def recover(self, model):
        tokens = self.get_tokens(threads=threads)
        tokens = set(itertools.chain.from_iterable(tokens))

        embeddings = []

        for text in tokens:
            embeddings.append([model.wv[token] for token in text if token in model.wv])

        return embeddings
