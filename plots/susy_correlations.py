import pandas as pd 
import numpy as np 
import tomotopy as tp 

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
import seaborn as sns

import textwrap


def reverse_colourmap(cmap, name):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = matplotlib.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

orig_cmap = matplotlib.cm.RdBu

mdl = tp.CTModel.load("output/hep-ct-75-0.1-0.001-130000-20/model")
correlations = mdl.get_correlations()

usages = pd.read_csv('output/supersymmetry_usages.csv')
usages = usages.groupby("term").agg(topic=("topic", lambda x: x.tolist()))
topics = usages.loc["supersymmetry"]["topic"] + [t for t in usages.loc["susy"]["topic"] if t not in usages.loc["supersymmetry"]["topic"]]

descriptions = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")
labels =  descriptions.loc[topics]["description"].tolist()

submatrix = correlations[topics,:][:,topics]

for i in range(submatrix.shape[0]):
    submatrix[i,i] = np.nan

w = textwrap.TextWrapper(width=20,break_long_words=False,replace_whitespace=False)
wlabels = ["\n".join(words) for words in map(w.wrap, labels)]

# shrunk_cmap = shiftedColorMap(orig_cmap, start=-submatrix.max(), midpoint=0, stop=+submatrix.max(), name='shrunk')
reverse_cmap = reverse_colourmap(orig_cmap, "reverse")
sns.heatmap(submatrix, xticklabels = wlabels, yticklabels = wlabels, annot=True, fmt=".2f", cmap=reverse_cmap, center=0, vmin=-1, vmax=1)

plt.savefig("plots/susy_correlations.pdf", bbox_inches="tight")
plt.savefig("plots/susy_correlations.pgf", bbox_inches="tight")
plt.savefig("plots/susy_correlations.eps", bbox_inches="tight")

