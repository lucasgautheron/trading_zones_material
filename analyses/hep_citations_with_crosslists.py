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
articles = articles[articles.categories.map(len)>0]
articles["cross_list"] = articles.categories.map(len)>1

for cat in hep_cats:
    articles[cat] = articles.categories.map(lambda cats: 1 if cat in cats else 0)

authors_references = pd.read_parquet("inspire-harvest/database/articles_authors.parquet")
authors = authors_references.merge(articles[["article_id"] + list(hep_cats.keys())], how="inner", left_on="article_id", right_on="article_id")
authors = authors.groupby("bai").agg(**{
    cat.replace("-", "_"): (cat, "sum") for cat in hep_cats
})
print(authors)
authors = authors[authors.sum(axis=1)>=3]
print(authors)
primary_category = authors.idxmax(axis=1).str.replace("_","-")

primary_category.to_csv("output/authors_primary_category.csv")

articles = articles.merge(authors_references, how="left", left_on="article_id", right_on="article_id")
articles = articles.merge(authors, how="left", left_on="bai", right_on="bai")

d = {
    "categories": ("categories", "first"),
    "date_created": ("date_created", "first")
}
d.update({
    cat.replace("-", "_"): (cat, "sum")
    for cat in hep_cats
})
articles = articles.groupby(["article_id"]).agg(**d).reset_index()

def decision_function(row):
    if len(row["categories"])==1:
        print("ok")
        return hep_cats[row["categories"][0]]
    else:
        contribs = np.array([row[cat.replace("-", "_")] for cat in hep_cats])
        most_frequent = np.argmax(contribs)
        tie = np.count_nonzero((contribs == most_frequent).astype(int))>1
        print(most_frequent, tie)
        return most_frequent if not tie else -1

    
print(articles)
articles["category"] = articles.apply(decision_function, axis=1)

articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(int)
articles = articles[(articles["year"] >= 1980) & (articles["year"] < 2020)]

print(articles.groupby(["year", "category"]).count())

articles = articles[articles["category"]>=0]


references = pd.read_parquet("inspire-harvest/database/articles_references.parquet")
references = references.merge(articles[["article_id", "category", "year"]], how='inner', left_on="cited", right_on="article_id")
articles = articles.merge(references, how='inner', left_on='article_id', right_on='cites', suffixes = ("_cites", "_cited"))

articles.to_parquet("output/cross_citations_crosslists.parquet")