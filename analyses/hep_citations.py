import pandas as pd
import numpy as np

import networkx as nx

import random

hep_cats = {
    "Theory-HEP": 0,
    "Phenomenology-HEP": 1,
    "Experiment-HEP": 2
}

def hep_filter(categories: list):
    return list(set(categories)&set(hep_cats.keys()))


articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "categories", "date_created"]]
articles["categories"] = articles.categories.map(hep_filter)
articles = articles[articles.categories.map(len)==1]
articles["category"] = articles.categories.map(lambda x: hep_cats[x[0]])

articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(int)
articles = articles[(articles["year"] >= 1980) & (articles["year"] < 2020)]

references = pd.read_parquet("inspire-harvest/database/articles_references.parquet")
references = references.merge(articles[["article_id", "category", "year"]], how='inner', left_on="cited", right_on="article_id")
articles = articles.merge(references, how='inner', left_on='article_id', right_on='cites', suffixes = ("_cites", "_cited"))

articles.to_parquet("output/cross_citations.parquet")