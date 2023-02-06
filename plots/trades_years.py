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

import seaborn as sns

from sklearn import linear_model

citations_th_ph = pd.read_csv(f"output/trading_zone_0_1/citations.csv")
citations_ph_th = pd.read_csv(f"output/trading_zone_1_0/citations.csv")
citations_th_ph["year_cites"] += 2001
citations_ph_th["year_cites"] += 2001

citations = pd.read_parquet("output/cross_citations.parquet")[["article_id_cited", "year_cited"]].drop_duplicates()
citations["article_id_cited"] = citations["article_id_cited"].astype(int)
citations_th_ph = citations_th_ph.merge(citations, left_on="article_id", right_on="article_id_cited")
citations_ph_th = citations_ph_th.merge(citations, left_on="article_id", right_on="article_id_cited")

citations_th_ph = citations_th_ph[citations_th_ph["year_cites"] >= citations_th_ph["year_cited"]]
citations_ph_th = citations_ph_th[citations_ph_th["year_cites"] >= citations_ph_th["year_cited"]]

p_t_th_ph = citations_th_ph.groupby(["year_cited", "year_cites"]).agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_th_ph["ratio"] = p_t_th_ph["trades"]/p_t_th_ph["total"]
p_t_th_ph_years = p_t_th_ph.groupby(["year_cites"]).agg(total_cites=("total","sum"))
p_t_th_ph = p_t_th_ph.merge(p_t_th_ph_years, left_index=True, right_index=True)
p_t_th_ph["ratio_time"] = p_t_th_ph["total"]/p_t_th_ph["total_cites"]

p_t_ph_th = citations_ph_th.groupby(["year_cited", "year_cites"]).agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_ph_th["ratio"] = p_t_ph_th["trades"]/p_t_ph_th["total"]
p_t_ph_th_years = p_t_ph_th.groupby(["year_cites"]).agg(total_cites=("total","sum"))
p_t_ph_th = p_t_ph_th.merge(p_t_ph_th_years, left_index=True, right_index=True)
p_t_ph_th["ratio_time"] = p_t_ph_th["total"]/p_t_ph_th["total_cites"]

ratio_ph_th = p_t_ph_th.reset_index()
ratio_ph_th = ratio_ph_th[ratio_ph_th["year_cited"]>=2001]

clf = linear_model.BayesianRidge()
clf.fit(ratio_ph_th[["year_cites"]], ratio_ph_th["ratio"])
print(clf.score(ratio_ph_th[["year_cites"]], ratio_ph_th["ratio"]))

clf = linear_model.BayesianRidge()
clf.fit(ratio_ph_th[["year_cited"]], ratio_ph_th["ratio"])
print(clf.score(ratio_ph_th[["year_cited"]], ratio_ph_th["ratio"]))

ratio_ph_th = ratio_ph_th.pivot(columns="year_cited", index="year_cites", values="ratio")

ratio_th_ph = p_t_th_ph.reset_index()
ratio_th_ph = ratio_th_ph[ratio_th_ph["year_cited"]>=2001]

print(ratio_th_ph["total"].describe())

clf = linear_model.BayesianRidge()
clf.fit(ratio_th_ph[["year_cites"]], ratio_th_ph["ratio"])
print(clf.score(ratio_th_ph[["year_cites"]], ratio_th_ph["ratio"]))

clf = linear_model.BayesianRidge()
clf.fit(ratio_th_ph[["year_cited"]], ratio_th_ph["ratio"])
print(clf.score(ratio_th_ph[["year_cited"]], ratio_th_ph["ratio"]))

ratio_th_ph = ratio_th_ph.pivot(columns="year_cited", index="year_cites", values="ratio")

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize = [6.4, 4.8*2])
sns.heatmap(ratio_th_ph, cmap="Reds", vmin=0, vmax=0.13, ax=axes[0])
axes[0].invert_yaxis()
axes[0].set(xlabel=None)
axes[0].set(xlabel=None)
axes[0].set_ylabel("citing paper publication year")
axes[0].set_title("$P($trade$)_{\\mathrm{th} \\to \\mathrm{ph}}$")

sns.heatmap(ratio_ph_th, cmap="Reds", vmin=0, vmax=0.13, ax=axes[1])
axes[1].invert_yaxis()
axes[1].set_xlabel("cited paper publication year")
axes[1].set_ylabel("citing paper publication year")
axes[1].set_title("$P($trade$)_{\\mathrm{ph} \\to \\mathrm{th}}$")
fig.savefig("plots/trading_zone_years.eps", bbox_inches="tight")

ratio = p_t_ph_th.reset_index()
ratio = ratio.pivot(columns="year_cited", index="year_cites", values="ratio_time")

plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize = [6.4, 4.8])

sns.heatmap(ratio, cmap="Reds", ax=ax, vmin=0, vmax=0.2)
ax.invert_yaxis()
ax.set_xlabel("cited paper year")
ax.set_ylabel("citing paper year")
plt.savefig("plots/trading_zone_years_time_1_0.eps", bbox_inches="tight")

ratio = p_t_th_ph.reset_index()
ratio = ratio.pivot(columns="year_cited", index="year_cites", values="ratio_time")

plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize = [6.4, 4.8])
sns.heatmap(ratio, cmap="Reds", ax=ax, vmin=0, vmax=0.2)
ax.invert_yaxis()
ax.set_xlabel("cited paper year")
ax.set_ylabel("citing paper year")
plt.savefig("plots/trading_zone_years_time_0_1.eps", bbox_inches="tight")
