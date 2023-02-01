import pandas as pd 

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

hep_cats = {"Theory-HEP", "Phenomenology-HEP", "Experiment-HEP", "Lattice"}
terms = sorted(["supersymmetry", "supersymmetric", "susy"])

articles = pd.read_parquet("inspire-harvest/database/articles.parquet")
articles["title"] = articles.title.str.lower()
articles["abstract"] = articles.abstract.str.lower()
articles = articles[articles["categories"].map(lambda x: hep_cats&set(x)).map(len) > 0]
articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(int)
articles = articles[(articles["year"] >= 1980) & (articles["year"] <= 2020)]

for cat in hep_cats:
    articles[cat] = articles["categories"].map(lambda x: cat in x)

cats = []

for term in terms:
    fit = articles[articles.title.str.contains(term) | articles.abstract.str.contains(term)]

    c = {}
    for cat in hep_cats-{"Lattice"}:
        c[cat] = fit[cat].mean()
    
    c["term"] = term

    cats.append(c)

cats = pd.DataFrame(cats)

rows = 1
cols = len(terms)

fig, axes = plt.subplots(rows, cols, sharey=True)

n = 0
for term in terms:
    ax = axes[n]
    
    labels = ["Theory-HEP", "Phenomenology-HEP", "Experiment-HEP"]
    human_friendly_labels = ["Theory", "Phenomenology", "Experiment"]
    probs = cats[cats["term"] == term][labels].iloc[0]

    if n == 0:
        ax.set_ylabel("Share of abstracts containing the term that belong to each category")
    
    ax.bar(human_friendly_labels, probs, color = ['#377eb8', '#ff7f00', '#4daf4a'])
    ax.set_title(f"``{term}''")
    ax.set_xticklabels(human_friendly_labels, rotation=90, ha="right")
    ax.set_label("Categories")

    n += 1

plt.subplots_adjust(bottom=0.2)
plt.savefig("plots/terms_cats.pgf", bbox_inches="tight")
plt.savefig("plots/terms_cats.pdf", bbox_inches="tight")