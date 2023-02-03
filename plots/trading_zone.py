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
parser.add_argument("--cited", choices=["0","1","2"], help="Cited category (0: theory, 1: phenomenology, 2: experiment)", required=True)
parser.add_argument("--cites", choices=["0","1","2"], help="Citing category (0: theory, 1: phenomenology, 2: experiment)", required=True)
args = parser.parse_args()

boundary = f"{args.cited}_{args.cites}"

ngrams = pd.read_csv(f"output/trading_zone_{boundary}/selected_ngrams.csv")
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
ngrams["drop"] = False

n_items = bow.shape[0]
n_words = bow.shape[1]

num = np.outer(bow[:2000].sum(axis=0),bow[:2000].sum(axis=0))/(2000**2)
den = np.tensordot(bow[:2000,:], bow[:2000,:], axes=([0],[0]))/2000
npmi = np.log(num)/np.log(den)-1

x, y = np.where(npmi-np.identity(n_words)>0.9)
for k,_ in enumerate(x):
    i = x[k]
    j = y[k]

    a = ngrams.at[i,"p_w_t_max"]
    b = ngrams.at[j,"p_w_t_max"]

    if a < b:
        ngrams.at[i,"drop"] = True
    else:
        ngrams.at[j,"drop"] = True

ngrams.sort_values("p_w_t_max", inplace=True)
ngrams = ngrams[ngrams["drop"]==False]

years = 2001+np.arange(len(p_t))

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
    axes[i].set_xlim(2001,2019)
    axes[i].set_ylim(0.003,0.3)
    axes[i].set_yscale("log")
    axes[i].legend(loc=("best" if i < 2 else "lower right"), prop={'size': 6})

plt.savefig(f"plots/trading_zone_{boundary}.pdf", bbox_inches="tight")
plt.savefig(f"plots/trading_zone_{boundary}.pgf", bbox_inches="tight")
plt.savefig(f"plots/trading_zone_{boundary}.eps", bbox_inches="tight")
