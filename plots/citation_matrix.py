import pandas as pd
import numpy as np

import networkx as nx

import random

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "xelatex",
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def is_hep(categories: str):
    return any(["-HEP" in x for x in categories])

articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "categories", "date_created", "abstract", "title"]]
articles["is_hep"] = articles.categories.map(is_hep)

articles = articles[articles["is_hep"]]
articles["th"] = articles.categories.map(lambda l: "Theory-HEP" in l)
articles["exp"] = articles.categories.map(lambda l: "Experiment-HEP" in l)
articles["ph"] = articles.categories.map(lambda l: "Phenomenology-HEP" in l)

articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(int)
articles = articles[(articles["year"] >= 2011) & (articles["year"] < 2020)]

references = pd.read_parquet("inspire-harvest/database/articles_references.parquet")
references = references.merge(articles[["article_id", "th", "exp", "ph"]], how='inner', left_on="cited", right_on="article_id")
articles = articles.merge(references, how='inner', left_on='article_id', right_on='cites', suffixes = ("_cites", "_cited"))

selected_articles = articles#[articles["article_id_cited"].isin(random.sample(set(articles["article_id_cited"].unique()), 10000))]

groups = ['exp', 'ph', 'th']
friendly_groups = ["Experiment", "Phenomenology", "Theory"]
indices = {groups[i]: i for i in range(len(groups))}

cites_matrix = np.zeros((len(groups),len(groups)))
cited_matrix = np.zeros((len(groups),len(groups)))
counts_cites = np.zeros(len(groups))
counts_cited = np.zeros(len(groups))

print("Building citation matrix")

for cited, cites in selected_articles.groupby("article_id_cited"):
    for c in cites.to_dict(orient="records"):
        w_cites = 1/(int(c["exp_cites"])+int(c["ph_cites"])+int(c["th_cites"]))
        w_cited = 1/(int(c["exp_cited"])+int(c["ph_cited"])+int(c["th_cited"]))
        for i in range(len(indices)):
            for j in range(len(indices)):
                if c[f"{groups[i]}_cites"] and c[f"{groups[j]}_cited"]:
                    cites_matrix[i,j] += w_cited*w_cites
                    cited_matrix[j,i] += w_cites*w_cited
                    counts_cites[i] += w_cited*w_cites
                    counts_cited[j] += w_cites*w_cited

sns.heatmap((cites_matrix/counts_cites.reshape(-1,1)).transpose(), cmap="Reds", annot=True, fmt=".2f", xticklabels = friendly_groups, yticklabels = friendly_groups, vmin=0, vmax=1)
plt.xlabel("Citing article's category")
plt.ylabel("Cited article's category")
plt.savefig("plots/cites_matrix.pgf")
plt.savefig("plots/cites_matrix.pdf")
plt.savefig("plots/cites_matrix.eps")

plt.clf()

sns.heatmap((cited_matrix/counts_cited.reshape(-1,1)).transpose(), cmap="Reds", annot=True, fmt=".2f", xticklabels = friendly_groups, yticklabels = friendly_groups, vmin=0, vmax=1)
plt.xlabel("Cited article's category")
plt.ylabel("Citing article's category")
plt.savefig("plots/cited_matrix.pgf")
plt.savefig("plots/cited_matrix.pdf")
plt.savefig("plots/cited_matrix.eps")
