#!/usr/bin/env python
from AbstractSemantics.terms import TermExtractor
import pandas as pd
import numpy as np
from os.path import join as opj

from collections import defaultdict

import re

import argparse
import yaml
import sys

def is_hep(categories: str):
    return any(["-HEP" in x for x in categories])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('trading zone')
    parser.add_argument('--location', help='model directory', default="output/trading_zone")
    parser.add_argument('--filter', choices=['categories', 'keywords', 'no-filter'], help='filter type', default="categories")
    parser.add_argument('--values', nargs='+', default=["Theory-HEP", "Phenomenology-HEP", "Experiment-HEP"], help='filter allowed values')
    parser.add_argument('--exclude', nargs='+', default=[], help='exclude values')
    parser.add_argument('--samples', type=int, default=10000000)
    parser.add_argument('--constant-sampling', type=int, default=0)
    parser.add_argument('--reuse-articles', default=False, action="store_true", help="reuse article selection")
    parser.add_argument('--nouns', default=False, action="store_true", help="include nouns")
    parser.add_argument('--adjectives', default=False, action="store_true", help="include adjectives")
    parser.add_argument('--lemmatize', default=True, action="store_true", help="stemmer")
    parser.add_argument('--lemmatize-ngrams', default=True, action="store_true", help="stemmer")
    parser.add_argument('--remove-latex', default=True, action="store_true", help="remove latex")
    parser.add_argument('--limit-redundancy', default=False, action="store_true", help="limit redundancy")
    parser.add_argument('--add-title', default=True, action="store_true", help="include title")
    parser.add_argument('--top-unithood', type=int, default=2000, help='top unithood filter')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--category-cited', type=int, help="filter cited category (0=theory,1=phenomenology,2=experiment)")
    parser.add_argument('--category-cites', type=int, help="filter citing category (0=theory,1=phenomenology,2=experiment)")
    parser.add_argument('--include-crosslists', default=False, action="store_true", help="include crosslists papers")
    args = parser.parse_args()

    location = f"{args.location}_{args.category_cited}_{args.category_cites}"

    with open(opj(location, "params.yml"), "w+") as fp:
        yaml.dump(args, fp)

    articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["title", "abstract", "article_id", "date_created", "categories"]]

    articles = articles[articles.categories.map(is_hep)]
    articles = articles[articles["date_created"].str.len() >= 4]
    articles["year"] = articles["date_created"].str[:4].astype(int)
    articles = articles[articles["year"]<=2019]

    if args.add_title:
        articles["abstract"] = articles["abstract"].str.cat(articles["title"])

    articles.drop(columns = ["title"], inplace=True)

    if args.remove_latex:
        articles['abstract'] = articles['abstract'].apply(lambda s: re.sub('$[^>]+$', '', s))

    articles = articles[articles["abstract"].map(len)>=100]
    articles["abstract"] = articles["abstract"].str.lower()

    articles = articles[articles["date_created"].str.len() >= 4]
    articles = articles[~articles["abstract"].isnull()]

    if args.constant_sampling > 0:
        articles = articles.groupby("year").head(args.constant_sampling)

    keep = pd.Series([False]*len(articles), index=articles.index)

    print("Applying filter...")
    if args.filter == 'keywords':
        for value in args.values:
            keep |= articles["abstract"].str.contains(value)
        for value in args.exclude:
            keep &= ~articles["abstract"].str.contains(value)
    elif args.filter == 'categories':
        for value in args.values:
            keep |= articles["categories"].apply(lambda l: value in l)
        for value in args.exclude:
            keep &= ~articles["categories"].apply(lambda l: value in l)

    articles = articles[keep==True]

    citations = articles[[x for x in articles.columns if x != "abstract"]].merge(pd.read_parquet("output/cross_citations{}.parquet".format("_crosslists" if args.include_crosslists else ""))[["article_id_cited", "article_id_cites", "category_cites", "category_cited", "year_cites", "year_cited"]], how="inner", left_on="article_id",right_on="article_id_cited")
    citations = citations[
        (citations["category_cited"] == args.category_cited) & (citations["category_cites"].isin([args.category_cites,args.category_cited]))
    ]
    citations["trade"] = (citations["category_cited"] != citations["category_cites"])
    citations = citations[citations["year_cites"]>=2001]
    citations = citations[citations["year_cites"]<=2019]
    citations["year_cites"] = ((citations["year_cites"]-citations["year_cites"].min())).astype(int)
    citations.drop_duplicates(["article_id_cited", "article_id_cites"], inplace=True)
    citations = citations.sample(args.samples if args.samples < len(citations) else len(citations))

    citations = citations.groupby(["article_id_cited", "year_cites"]).agg(
        trades = ("trade", "sum"),
        total = ("article_id_cites", "count"),
        category_cited = ("category_cited", "first")
    )

    citations.reset_index(inplace=True)
    articles_to_keep = list(citations["article_id_cited"].unique())

    citations = citations[citations["article_id_cited"].isin(articles_to_keep)]        

    articles = articles[articles["article_id"].isin(articles_to_keep)]
    articles = articles.merge(
        citations[["article_id_cited", "category_cited"]].drop_duplicates(),
        left_on="article_id",
        right_on="article_id_cited"
    )


    print("Extracting n-grams...")
    extractor = TermExtractor(
        articles["abstract"].tolist(),
        limit_redundancy=args.limit_redundancy
    )

    if args.nouns:
        extractor.add_patterns([["NN.*"]])

    if args.adjectives:
        extractor.add_patterns([["^JJ$"]])

    ngrams = extractor.ngrams(
        threads=args.threads,
        lemmatize=args.lemmatize,
        lemmatize_ngrams=args.lemmatize_ngrams
    )
    ngrams = map(lambda l: [" ".join(n) for n in l], ngrams)
    ngrams = list(ngrams)

    articles["ngrams"] = ngrams

    print("Deriving vocabulary...")
    ngrams_occurrences = defaultdict(int)        
    categories = articles["category_cited"].tolist()

    for ngrams in articles["ngrams"].tolist():            
        _ngrams = set(ngrams)
        for ngram in _ngrams:
            ngrams_occurrences[ngram] += 1
        
    ngrams_occurrences = {
        "ngram": ngrams_occurrences.keys(),
        "count": ngrams_occurrences.values()
    }
    
    ngrams_occurrences = pd.DataFrame(ngrams_occurrences)
    
    ngrams_occurrences["unithood"] = (
        np.log(2 + ngrams_occurrences["ngram"].str.count(" "))
        * ngrams_occurrences["count"] / len(articles)
    )
        
    ngrams_occurrences.set_index("ngram", inplace=True)

    top = ngrams_occurrences.sort_values("unithood", ascending=False).head(
        args.top_unithood
    )
    top.to_csv(opj(location, "ngrams.csv"))

    articles = articles.sample(frac=1)
    ngrams = articles["ngrams"].tolist()

    selected_ngrams = pd.read_csv(opj(location, 'ngrams.csv'))['ngram'].tolist()

    vocabulary = {
        n: i
        for i, n in enumerate(selected_ngrams)
    }

    ngrams = [[ngram for ngram in _ngrams if ngram in selected_ngrams] for _ngrams in ngrams]
    ngrams_bow = [[vocabulary[ngram] for ngram in _ngrams] for _ngrams in ngrams]
    ngrams_bow = [[_ngrams.count(i) for i in range(len(selected_ngrams))] for _ngrams in ngrams_bow]

    n = []
    frac = []

    ngrams_bow = np.array(ngrams_bow)
    for i in range(0, args.top_unithood, 10):
        sum_bow = ngrams_bow[:,:i]
        sum_bow = sum_bow.sum(axis=1)
        sum_bow = (sum_bow==0).mean()
        n.append(i)
        frac.append(100*sum_bow)

    frac = np.array(frac)
    n_words = n[np.argmin(frac[frac>5])]
    
    print(f"preserving {n_words} words out of {len(selected_ngrams)}.")

    selected_ngrams = selected_ngrams[:n_words]
    
    vocabulary = {
        n: i
        for i, n in enumerate(selected_ngrams)
    }

    ngrams = [[ngram for ngram in _ngrams if ngram in selected_ngrams] for _ngrams in ngrams]
    ngrams_bow = [[vocabulary[ngram] for ngram in _ngrams] for _ngrams in ngrams]
    ngrams_bow = [[_ngrams.count(i) for i in range(len(selected_ngrams))] for _ngrams in ngrams_bow]

    full_bow = (np.array(ngrams_bow)>=1)*1
    np.save(opj(location, 'full_bow.npy'), full_bow)
 
    pd.DataFrame({'ngram': selected_ngrams}).to_csv(opj(location, "selected_ngrams.csv"), index=False)
    
    articles.reset_index(inplace=True)
    articles["id"] = articles.index+1
    citations = citations.merge(articles[["article_id", "id"]], left_on="article_id_cited",right_on="article_id")
    citations.to_csv(opj(location, 'citations.csv'))
