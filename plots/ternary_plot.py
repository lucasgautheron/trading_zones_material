import matplotlib.tri as tri
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import argparse

mpl.use("pgf")
mpl.rcParams.update(
    {
        "pgf.texsystem": "xelatex",
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

parser = argparse.ArgumentParser()
parser.add_argument('fit')
args = parser.parse_args(["output/social_divide_samples.parquet"])

df = pd.read_parquet(args.fit)
data = {'n_authors': 2500, 'n_types': 3}

cats = ['th','exp','ph']

n_authors = data['n_authors']
scale = 1
# n_authors = 1547-1

a = [df[f'probs.{k}.1'].mean() for k in 1+np.arange(n_authors)]
b = [df[f'probs.{k}.3'].mean() for k in 1+np.arange(n_authors)]
c = [df[f'probs.{k}.2'].mean() for k in 1+np.arange(n_authors)]

a = np.array(a).flatten()
b = np.array(b).flatten()
c = np.array(c).flatten()

s = (a+b+c)/scale 
a = a/s
b = b/s
c = c/s

figure, tax = ternary.figure(scale=scale)
tax.boundary(linewidth=1)
tax.gridlines(linewidth=0.25, multiple=scale/5, linestyle="-.", color='black')
tax.gridlines(linewidth=0.1, multiple=scale/10, linestyle="-.", color='black')
tax.ticks(axis='lbr', linewidth=1, multiple=scale/5, tick_formats="%.1f")

fontsize = 10

tax.right_corner_label("Théorie", fontsize=fontsize)
tax.top_corner_label("Phénoménologie", fontsize=fontsize)
tax.left_corner_label("Expérience", fontsize=fontsize)

tax.left_axis_label("$\\frac{p_{exp}}{p_{exp}+p_{ph}+p_{th}}$", fontsize=fontsize, offset=0.14)
tax.right_axis_label("$\\frac{p_{ph}}{p_{exp}+p_{ph}+p_{th}}$", fontsize=fontsize, offset=0.14)
tax.bottom_axis_label("$\\frac{p_{th}}{p_{exp}+p_{ph}+p_{th}}$", fontsize=fontsize)

tax.scatter(np.array([a,b,c]).T, s=0.1, alpha=0.5, color='red')

tax.get_axes().axis('off')
tax.clear_matplotlib_ticks()

tax.set_title(f"Fraction of (co-)authored publications for each category (theory, phenomenology, experiment)", y=1.1)

x0 = 0.32
y0 = 0.825
dx = 0.1/2
dy = -0.1*np.sqrt(3)/2

plt.arrow(x0, y0, dx, dy, head_width=0.01, head_length=0.01, fc='k', ec='k')

x0 = 1.075
y0 = 0.1725
dx = -0.1
dy = 0

plt.arrow(x0, y0, dx, dy, head_width=0.01, head_length=0.01, fc='k', ec='k')


x0 = 0.2-0.16/2
y0 = -0.16*np.sqrt(3)/2
dx = 0.1/2
dy = 0.1*np.sqrt(3)/2

plt.arrow(x0, y0, dx, dy, head_width=0.01, head_length=0.01, fc='k', ec='k')

plt.savefig(f"plots/social_divide_ternary.pgf", bbox_inches='tight')
plt.savefig(f"plots/social_divide_ternary.pdf", bbox_inches='tight')