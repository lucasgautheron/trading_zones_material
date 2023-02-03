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

citations_th_ph = pd.read_csv(f"output/trading_zone_0_1/citations.csv")
citations_ph_th = pd.read_csv(f"output/trading_zone_1_0/citations.csv")
citations_exp_ph = pd.read_csv(f"output/trading_zone_2_1/citations.csv")
citations_ph_exp = pd.read_csv(f"output/trading_zone_1_2/citations.csv")

p_t_th_ph = citations_th_ph.groupby("year_cites").agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_th_ph = p_t_th_ph["trades"]/p_t_th_ph["total"]

p_t_ph_th = citations_ph_th.groupby("year_cites").agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_ph_th = p_t_ph_th["trades"]/p_t_ph_th["total"]

p_t_exp_ph = citations_exp_ph.groupby("year_cites").agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_exp_ph = p_t_exp_ph["trades"]/p_t_exp_ph["total"]

p_t_ph_exp = citations_ph_exp.groupby("year_cites").agg(
    trades = ("trades", "sum"),
    total = ("total", "sum")
)
p_t_ph_exp = p_t_ph_exp["trades"]/p_t_ph_exp["total"]

years = p_t_th_ph.index.values+2001

print(years.shape)
print(p_t_th_ph.shape)

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
colors += colors

fig, ax = plt.subplots(1,1,sharey=True)

ax.scatter(years, p_t_th_ph, color=colors[0], s=10, label="$P($phenomenological $\\to$ theoretical$)$")
ax.plot(years, p_t_th_ph, color=colors[0], lw=0.5)
ax.scatter(years, p_t_ph_th, color=colors[1], s=10, label="$P($theoretical $\\to$ phenomenological$)$")
ax.plot(years, p_t_ph_th, color=colors[1], lw=0.5)

# ax.scatter(years, p_t_exp_ph, color=colors[2], s=10, label="phenomenological $\\to$ experimental")
# ax.plot(years, p_t_exp_ph, color=colors[2], lw=0.5)
# ax.scatter(years, p_t_ph_exp, color=colors[3], s=10, label="experimental $\\to$ phenomenological")
# ax.plot(years, p_t_ph_exp, color=colors[3], lw=0.5)

ax.set_ylabel("$P($trade$)$")

ax.set_xticks(np.arange(2000,2020,2))
ax.set_xlim(2001,2019)
ax.set_ylim(0,0.15)
ax.legend()

plt.savefig(f"plots/trades.pdf", bbox_inches="tight")
plt.savefig(f"plots/trades.pgf", bbox_inches="tight")

plt.cla()
plt.clf()
