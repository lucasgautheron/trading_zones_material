import pandas as pd 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt 
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
plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\setmainfont{amssymb}",
])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cited", choices=["0","1","2"], help="Cited category (0: theory, 1: phenomenology, 2: experiment)")
parser.add_argument("--cites", choices=["0","1","2"], help="Citing category (0: theory, 1: phenomenology, 2: experiment)")
args = parser.parse_args()

boundary = f"{args.cited}_{args.cites}"

articles = pd.read_parquet("../inspire-harvest/database/articles.parquet")[["date_created", "categories"]]

articles = articles[articles.categories.map(lambda l: "Theory-HEP" in l or "Phenomenology-HEP" in l)]
articles = articles[articles["date_created"].str.len() >= 4]
articles["year"] = articles["date_created"].str[:4].astype(int)-2001
articles = articles[(articles["year"] >= 0) & (articles["year"] <= 19)]
articles["year"] = (articles["year"]/2).astype(int)
articles["ph"] = articles["categories"].map(lambda l: "Phenomenology-HEP" in l)

articles = articles.groupby("year").agg(
    ph = ('ph', 'mean')
).reset_index()

articles["year"] = 2001+articles["year"]

ngrams = pd.read_csv(f"output/trading_zone_{boundary}/ngrams.csv")
ngrams["keyword"] = ngrams.index+1
supersymmetry_keywords = ngrams[ngrams["ngram"].str.contains("super")]["keyword"].tolist()
n_ngrams = len(ngrams)

citations = pd.read_csv(f"output/trading_zone_{boundary}/citations.csv")
p_t = citations.groupby("year_cites")["trades"].sum().values

bow = np.load(f"output/trading_zone_{boundary}/full_bow.npy")
p_w_t = np.zeros((len(p_t), bow.shape[1]))

for citation in citations.to_dict(orient="records"):
    i = citation["id"]-1
    p_w_t[citation["year_cites"],:] += bow[i,:]*citation["trades"]

p_w_t = (p_w_t.T/p_t).T
ngrams["p_w_t_max"] = p_w_t.max(axis=0)
ngrams.sort_values("p_w_t_max", inplace=True)

years = 2000+np.arange(len(p_t))

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
colors += colors

fig, axes = plt.subplots(1,2,sharey=True)

ax = axes[0]
n = 0
for i, ngram in ngrams.tail(5).to_dict(orient="index").items():
    p = p_w_t[:,i]
    ax.scatter(years, p, color=colors[n], s=10, label=ngram["ngram"])
    ax.plot(years, p, color=colors[n], lw=0.5)
    ax.set_ylabel(f"$P(b_k=1|$trade$)$")

    n += 1

ax = axes[1]
n = 0
for i, ngram in ngrams[ngrams["ngram"].str.contains("super")].tail(5).to_dict(orient="index").items():
    if n >= len(colors):
        break

    p = p_w_t[:,i]
    ax.scatter(years, p, color=colors[n], s=10, label=ngram["ngram"])
    ax.plot(years, p, color=colors[n], lw=0.5)
    ax.set_ylabel(f"$P(b_k=1|$trade$)$")

    n += 1

for i in range(2):
    axes[i].set_xlim(2001,2020)
    # axes[i].set_ylim(0,4)
    axes[i].set_ylim(0.003,0.3)
    axes[i].set_yscale("log")
    axes[i].plot([2000,2020.5], [1, 1], ls="-", color="black")
    axes[i].legend(loc=("best" if i < 2 else "lower right"), prop={'size': 6})

plt.savefig(f"plots/trading_zone_{boundary}.pdf", bbox_inches="tight")
plt.savefig(f"plots/trading_zone_{boundary}.pgf", bbox_inches="tight")
