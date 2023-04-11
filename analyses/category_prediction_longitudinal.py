from AbstractSemantics.terms import TermExtractor
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CT Model')
    parser.add_argument('location', help='model directory')
    parser.add_argument('filter', choices=['categories', 'keywords', 'no-filter'], help='filter type')
    parser.add_argument('--values', nargs='+', default=[], help='filter allowed values')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--constant-sampling', type=int, default=0)
    parser.add_argument('--reuse-articles', default=False, action="store_true", help="reuse article selection")
    parser.add_argument('--nouns', default=False, action="store_true", help="include nouns")
    parser.add_argument('--adjectives', default=False, action="store_true", help="include adjectives")
    parser.add_argument('--lemmatize', default=False, action="store_true", help="stemmer")
    parser.add_argument('--lemmatize-ngrams', default=False, action="store_true", help="stemmer")
    parser.add_argument('--remove-latex', default=False, action="store_true", help="remove latex")
    parser.add_argument('--limit-redundancy', default=False, action="store_true", help="limit redundancy")
    parser.add_argument('--add-title', default=False, action="store_true", help="include title")
    parser.add_argument('--top-unithood', type=int, default=20000, help='top unithood filter')
    parser.add_argument('--min-token-length', type=int, default=0, help='minimum token length')
    parser.add_argument('--min-df', type=int, default=0, help='min_df')
    parser.add_argument('--reuse-stored-vocabulary', default=False, action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(["output/category_prediction_longitudinal", "categories", "--values", "Phenomenology-HEP", "Theory-HEP", "--samples", "400000", "--nouns", "--lemmatize", "--lemmatize-ngrams", "--remove-latex", "--add-title", "--top-unithood", "1000", "--threads", "16"])

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
    extractor = TermExtractor(articles["abstract"].tolist(), limit_redundancy=args.limit_redundancy)

    if args.nouns:
        extractor.add_patterns([["NN.*"]])

    if args.adjectives:
        extractor.add_patterns([["^JJ$"]])

    ngrams = extractor.ngrams(threads=args.threads,lemmatize=args.lemmatize,lemmatize_ngrams=args.lemmatize_ngrams)
    ngrams = map(lambda l: [" ".join(n) for n in l], ngrams)
    ngrams = list(ngrams)

    articles["ngrams"] = ngrams

    print("n_articles:", len(articles))

    print("Deriving vocabulary...")
    if not args.reuse_stored_vocabulary:
        ngrams_occurrences = defaultdict(int)

        for ngrams in articles["ngrams"].tolist():
            _ngrams = set(ngrams)
            for ngram in _ngrams:
                ngrams_occurrences[ngram] += 1

        ngrams_occurrences = pd.DataFrame(
            {"ngram": ngrams_occurrences.keys(), "count": ngrams_occurrences.values()}
        )
        ngrams_occurrences["unithood"] = (
            np.log(2 + ngrams_occurrences["ngram"].str.count(" "))
            * ngrams_occurrences["count"]
        )
        ngrams_occurrences["unithood"] /= len(articles)
        ngrams_occurrences.set_index("ngram", inplace=True)

        ngrams_occurrences["len"] = ngrams_occurrences.index.map(len)
        ngrams_occurrences = ngrams_occurrences[ngrams_occurrences["len"] > 1]

        top = ngrams_occurrences.sort_values("unithood", ascending=False).head(
            args.top_unithood
        )

        top.to_csv(opj(args.location, "ngrams.csv"))

    
    selected_ngrams = pd.read_csv(opj(args.location, 'ngrams.csv'))['ngram'].tolist()

    vocabulary = {
        n: i
        for i, n in enumerate(selected_ngrams)
    }

    inv_vocabulary = {
        vocabulary[v]: v
        for v in vocabulary
    }

    ngrams = articles["ngrams"].tolist()
    ngrams = [[ngram for ngram in _ngrams if ngram in selected_ngrams] for _ngrams in ngrams]

    bow = [[vocabulary[ngram] for ngram in _ngrams] for _ngrams in ngrams]
    bow = [[_ngrams.count(i) for i in range(len(selected_ngrams))] for _ngrams in bow]
    bow = np.array(bow)
    bow = (bow>0)*1 # destroy freq information

    tfidf = TfidfTransformer()
    bow_tfidf = tfidf.fit_transform(bow).todense().tolist()
    articles["bow_tfidf"] = bow_tfidf

    cat_classifier = MultiLabelBinarizer(sparse_output=False)
    articles["categories"] = articles["categories"].map(lambda l: list(set(l)&{"Phenomenology-HEP", "Theory-HEP"}))
    cats = cat_classifier.fit_transform(articles["categories"]).tolist()
    articles["cats"] = cats

    vocab = 500

    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score

    results = []

    for year_group, train in articles.groupby("year_group"):
        train = articles[articles["year_group"] != year_group]
        
        for i in range(2):
            fit = LogisticRegression(random_state=0,max_iter=200).fit(np.stack(train["bow_tfidf"].values)[:,0:vocab], np.stack(train["cats"].values).astype(int)[:,i])

            for j in range(vocab):
                results.append({
                    'year_group': year_group,
                    'term': inv_vocabulary[j],
                    'category': cat_classifier.inverse_transform(np.array([np.identity(2)[i,:]]))[0][0],
                    'coef': fit.coef_[0,j],
                    'rank': j
                })


    results = pd.DataFrame(results)
    results["drop"] = False

    bow = (bow>=1).astype(int)
    num = np.outer(bow[:3000,:vocab].sum(axis=0),bow[:3000,:vocab].sum(axis=0))/(3000**2)
    den = np.tensordot(bow[:3000,:vocab], bow[:3000,:vocab], axes=([0],[0]))/3000
    npmi = np.log(num)/np.log(den)-1

    x, y = np.where(npmi-np.identity(vocab)>=0.95)
    for k,_ in enumerate(x):
        i = x[k]
        j = y[k]

        a = inv_vocabulary[i]
        b = inv_vocabulary[j]

        if (not (a in b or b in a)):
            continue

        if i > j:
            results.loc[results['rank'] == i, 'drop'] = True
        else:
            results.loc[results['rank'] == j, 'drop'] = True

    results = results[results["drop"]==False]
    results = results[results["term"].str.match("^[a-zA-Z--- ]*$")]
    results.sort_values(["year_group", "rank"]).to_csv(opj(args.location, "results.csv"))
