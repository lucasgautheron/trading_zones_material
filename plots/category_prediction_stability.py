import pandas as pd 
import numpy as np 

from scipy.stats import binom

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

cats = ["Experiment", "Phenomenology", "Theory"]
colors = ['#4daf4a', '#ff7f00', '#377eb8']

accuracy = pd.read_csv("output/category_prediction/accuracy_per_period.csv").sort_values("year_group")
accuracy = accuracy[accuracy["year_group"]<8]

fig, ax = plt.subplots(1,1)
for i in range(3):
    ci = ([],[])
    for row in accuracy.to_dict(orient="records"):
        low,high = binom.ppf([0.025, 0.975], row[f"count_{i}"], row[f"accurate_{i}"], loc=0)
        ci[0].append(low/row[f"count_{i}"])
        ci[1].append(high/row[f"count_{i}"])

    ax.scatter(accuracy["year_group"], accuracy[f"accurate_{i}"], color=colors[i], label=cats[i])
    ax.errorbar(accuracy["year_group"], accuracy[f"accurate_{i}"], yerr=(ci[0]-accuracy[f"accurate_{i}"], accuracy[f"accurate_{i}"]-ci[1]), color=colors[i], ls="none")

    ax.plot(accuracy["year_group"], accuracy[f"dummy_accurate_{i}"], color=colors[i], ls="dashed")

ax.set_xticks(accuracy["year_group"].tolist())
ax.set_xticklabels([f"{1980+i*5}-{1980+(i+1)*5-1}" for i in accuracy["year_group"].tolist()], rotation=45, ha='right')

ax.set_ylabel("Accuracy")

ax.legend()
fig.savefig("plots/category_prediction_stability.eps", bbox_inches="tight")
fig.savefig("plots/category_prediction_stability.png", bbox_inches="tight")
plt.show()