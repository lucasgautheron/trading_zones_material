import pandas as pd 
import numpy as np
import textwrap

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
usages = pd.read_csv("output/supersymmetry_usages.csv")
descriptions = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")
usages = usages.merge(descriptions)

rows = 1
cols = usages["term"].nunique()

fig, axes = plt.subplots(rows, cols, sharey=True)

n = 0
for term, topics in usages.groupby("term"):
    topics = topics.sort_values("p_t_w", ascending=False)
    ax = axes[n]
    
    probs = topics["p_t_w"].values
    labels = topics["description"].tolist()

    w = textwrap.TextWrapper(width=40,break_long_words=False,replace_whitespace=False)
    wlabels = ["\n".join(words) for words in map(w.wrap, labels)]

    if n == 0:
        ax.set_ylabel("Probability $P(z|w)$ that $w$ occurs as part of a topic $z$")
    
    ax.bar(labels, probs)
    ax.set_title(f"``{term}''")
    ax.set_xticklabels(wlabels, rotation=60, ha="right")
    ax.set_label("Topics $z$")

    n += 1

plt.subplots_adjust(bottom=0.2)
plt.savefig("plots/susy_usages.pgf", bbox_inches="tight")
plt.savefig("plots/susy_usages.pdf", bbox_inches="tight")