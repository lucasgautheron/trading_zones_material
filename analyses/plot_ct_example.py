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
articles = pd.read_parquet("../inspire-harvest/database/articles.parquet")[["article_id", "title", "abstract"]]
topics = pd.read_parquet("output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet")
topics = topics.merge(articles, how="inner")
del articles
descriptions = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv").set_index("topic")

topics = topics[topics["title"].str.lower().str.contains("supersymmetr")]
topics = topics[topics["abstract"].map(len) >= 200]

topics["entropy"] = topics["probs"].apply(lambda x: -np.sum(np.log(x)*x))
topics.sort_values("entropy", ascending=True, inplace=True)
topics.reset_index(inplace=True)
topics["10_largest"] = topics["probs"].apply(lambda x: x.argsort()[-10:][::-1])

topics = topics.head(100).sample(frac=1)
topics["article_id"] = topics["article_id"].astype(int)
topics.set_index("article_id", inplace=True)
topics = topics.loc[[733150, 535075, 678629],:]

rows = 1
cols = 3

fig, axes = plt.subplots(rows, cols, sharey=True)

n = 0
for i in range(rows):
    for j in range(cols):
        ax = axes[j]
        
        article = topics.iloc[n]
        probs = article["probs"]
        title = article["title"]

        tw = textwrap.TextWrapper(width=20,break_long_words=False)
        title = "\n".join(tw.wrap(title))


        labels = map(lambda x: descriptions.iloc[x]["description_fr"], article["10_largest"])
        labels = list(labels)

        if n == 0:
            ax.set_ylabel("Contribution des principaux topics ($\\theta_{d,z}$)")
        
        ax.bar(labels[:5], np.take(probs, article["10_largest"][:5]))
        ax.set_title(f"``{title}''")
        ax.set_xticklabels(labels, rotation=60, ha="right")

        n += 1

plt.subplots_adjust(bottom=0.2)
plt.savefig("plots/example_articles_topic_dist.pgf", bbox_inches="tight")
plt.savefig("plots/example_articles_topic_dist.pdf", bbox_inches="tight")