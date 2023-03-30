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
from adjustText import adjust_text

import textwrap

from sklearn.manifold import TSNE

articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "pacs_codes", "categories"]]
articles["article_id"] = articles["article_id"].astype(int)

topics = pd.read_parquet("output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet")
topics["article_id"] = topics["article_id"].astype(int)
topics["topics"] = topics["probs"]
topics = topics.merge(articles, how="inner", left_on = "article_id", right_on = "article_id")
topics["categories"] = topics["categories"].map(
    lambda l: (
        [x in l for x in ["Theory-HEP", "Phenomenology-HEP", "Experiment-HEP"]]
    )
)

X = np.stack(topics["topics"].values)
Y = np.stack(topics["categories"].values).astype(int)

cat_topic_mean = np.zeros((Y.shape[1], X.shape[1]))
for i in range(3):
    cat_topic_mean[i] = X[Y[:,i]==1,:].mean(axis=0)

topic_main_category = cat_topic_mean.argmax(axis=0).astype(int)

usages = pd.read_csv('output/supersymmetry_usages.csv')
usages = usages.groupby("term").agg(topic=("topic", lambda x: x.tolist()))
susy_topics = usages.loc["supersymmetry"]["topic"] + [t for t in usages.loc["susy"]["topic"] if t not in usages.loc["supersymmetry"]["topic"]]

descriptions = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")
labels =  descriptions.loc[susy_topics]["description"].tolist()

edges = np.array([False]*len(topic_main_category))
edges[susy_topics]=True
edges = ["black" if edge else "none" for edge in edges]

mdl = tp.CTModel.load("output/hep-ct-75-0.1-0.001-130000-20/model")
correlations = mdl.get_correlations()


colors=['#377eb8', '#ff7f00', '#4daf4a']
cats=["Theory", "Phenomenology", "Experiment"]

tsne = TSNE(n_components=2, metric="precomputed", random_state=714, perplexity=40)
points = tsne.fit_transform(1-correlations)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(points, topic_main_category)
angle = np.arctan(reg.coef_[0]/reg.coef_[1])-np.pi/2
m = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle),np.cos(angle)]])
points=points@m

fig, axes = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={"height_ratios": [5, 1]})

for i, cat in enumerate(cats):
    axes[0].scatter(
        points[topic_main_category==i,0],
        points[topic_main_category==i,1],
        color=colors[i],
        label=cat,
        edgecolors=[edges[i[0]] for i in np.argwhere(topic_main_category==i) if i!=np.nan]
    )

texts = []

for i,topic in enumerate(susy_topics):
    texts.append(
        axes[0].annotate(
            labels[i],
            xy=(points[topic,0],points[topic,1]),
            xytext=(points[topic,0],points[topic,1]+0.25),
            size="small"
        )
    )

adjust_text(texts,ax=axes[0])

import seaborn as sns
sns.kdeplot(data=[points[topic_main_category==i,0] for i in range(3)],ax=axes[1],legend=False)

plt.subplots_adjust(wspace=0, hspace=0)

for i in range(2):
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])

axes[1].set_ylabel("")

axes[0].legend()
fig.savefig(f"plots/topics_tsne.eps", bbox_inches="tight")
fig.savefig(f"plots/topics_tsne.png", bbox_inches="tight")