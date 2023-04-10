from AbstractSemantics.terms import TermExtractor
from AbstractSemantics.embeddings import GensimWord2Vec
import pandas as pd
import numpy as np
from os.path import join as opj
from os.path import exists

import itertools
from functools import partial
from collections import defaultdict

import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

import multiprocessing as mp

from matplotlib import pyplot as plt

import argparse
import yaml
import sys

from gensim.models.callbacks import CallbackAny2Vec

class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        
        self.epoch += 1
        self.loss_previous_step = loss
        
        for word in self._test_words:  # show wv logic changes
            print(f"{word}: {model.wv.most_similar(word)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CT Model')
    parser.add_argument('location', help='model directory')
    parser.add_argument('filter', choices=['categories', 'keywords', 'no-filter'], help='filter type')
    parser.add_argument('--values', nargs='+', default=[], help='filter allowed values')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--dimensions', type=int, default=64)
    parser.add_argument('--constant-sampling', type=int, default=0)
    parser.add_argument('--reuse-articles', default=False, action="store_true", help="reuse article selection")
    parser.add_argument('--nouns', default=False, action="store_true", help="include nouns")
    parser.add_argument('--adjectives', default=False, action="store_true", help="include adjectives")
    parser.add_argument('--lemmatize', default=False, action="store_true", help="stemmer")
    parser.add_argument('--remove-latex', default=False, action="store_true", help="remove latex")
    parser.add_argument('--add-title', default=False, action="store_true", help="include title")
    parser.add_argument('--top-unithood', type=int, default=20000, help='top unithood filter')
    parser.add_argument('--min-token-length', type=int, default=0, help='minimum token length')
    parser.add_argument('--min-df', type=int, default=0, help='min_df')
    parser.add_argument('--reuse-stored-vocabulary', default=False, action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(["output/embeddings", "categories", "--values", "Phenomenology-HEP", "Theory-HEP", "--samples", "150000", "--threads", "4"])

    with open(opj(args.location, "params.yml"), "w+") as fp:
        yaml.dump(args, fp)

    articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["title", "abstract", "article_id", "date_created", "categories"]]

    if args.add_title:
        articles["abstract"] = articles["abstract"].str.cat(articles["title"])

    articles.drop(columns = ["title"], inplace=True)

    if args.remove_latex:
        articles['abstract'] = articles['abstract'].apply(lambda s: re.sub('$[^>]+$', '', s))

    articles = articles[articles["abstract"].map(len)>=100]
    articles["abstract"] = articles["abstract"].str.lower()

    articles = articles[articles["date_created"].str.len() >= 4]
    articles["year"] = articles["date_created"].str[:4].astype(int)-1980
    articles = articles[(articles["year"] >= 0) & (articles["year"] <= 40)]
    articles["year_group"] = articles["year"]//5

    if args.reuse_articles:
        used = pd.read_csv(opj(args.location, 'articles.csv'))
        articles = articles[articles["article_id"].isin(used["article_id"])]
    else:
        articles = articles[~articles["abstract"].isnull()]

        if args.constant_sampling > 0:
            articles = articles.groupby("year").head(args.constant_sampling)

        keep = pd.Series([False]*len(articles), index=articles.index)

        print("Applying filter...")
        if args.filter == 'keywords':
            for value in args.values:
                keep |= articles["abstract"].str.contains(value)
        elif args.filter == 'categories':
            for value in args.values:
                keep |= articles["categories"].apply(lambda l: value in l)

        articles = articles[keep==True]
        articles = articles.sample(frac=1).head(args.samples)
        articles[["article_id"]].to_csv(opj(args.location, 'articles.csv'))

    articles.reset_index(inplace = True)

    print("Extracting n-grams...")
    extractor = TermExtractor(articles["abstract"].tolist())
    sentences = extractor.tokens(threads=args.threads, lemmatize=True, split_sentences=True)

    print(len(sentences))
    print(sentences[0])
    print(sentences[0][0])

    articles["sentences"] = sentences

    for category in args.values:
        _articles = articles[articles.categories.map(lambda l: category in l)]

        corpus = [sentence for sentences in _articles["sentences"].tolist() for sentence in sentences]

        print(category, len(corpus))

        emb = GensimWord2Vec(corpus)
        model = emb.model(
            vector_size=args.dimensions,
            window=10,
            workers=args.threads,
            compute_loss=True,
            epochs=50,
            callbacks=[MonitorCallback(["quark", "gluino", "renormalization"])]
        )
        # model.build_vocab(corpus)
        model.train(corpus, epochs=10, total_examples=model.corpus_count)
        model.train(corpus, epochs=10, total_examples=model.corpus_count)
        model.train(corpus, epochs=10, total_examples=model.corpus_count)
        model.save(opj(args.location, f"{category}.mdl"))
