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

cats = {
    "Theory-HEP": "Théorie",
    "Phenomenology-HEP": "Phéno.",
    "Experiment-HEP": "Exp."
}

articles = pd.read_parquet("../inspire-harvest/database/articles.parquet")[["article_id", "title", "abstract", "categories"]]
topics = pd.read_parquet("output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet")
topics = topics.merge(articles, how="inner")
del articles
descriptions = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv").set_index("topic")

usages = pd.read_csv("analyses/supersymmetry_usages.csv")
susy_topics = list(usages["topic"].unique())

topics = topics[topics["title"].str.lower().str.contains("super") | topics["title"].str.lower().str.contains("susy")]
topics = topics[topics["abstract"].map(len) >= 500]

articles = []

for susy_topic in susy_topics:
    print(susy_topic)
    topics[f"susy_{susy_topic}"] = topics["probs"].map(lambda l: l[susy_topic])

    topics.sort_values(f"susy_{susy_topic}", ascending=False, inplace=True)

    articles.append(
        topics.head(3).assign(
            topic=susy_topic,
            description=descriptions.iloc[susy_topic]["description_fr"],
            prob=topics.head(3)[f"susy_{susy_topic}"]
        )
    )

articles = pd.concat(articles)
articles = articles[["article_id", "title", "description", "prob", "categories"]]
articles["categories"] = articles["categories"].map(lambda l: "/".join([cats[x] for x in list(set(l)&{"Theory-HEP", "Phenomenology-HEP", "Experiment-HEP"})]))
articles["prob"] = articles["prob"].map(lambda f: f"{f:.2f}")
articles.rename(columns = {
    "description": "Sujet",
    "title": "Article",
    "categories": "Catégories",
    "prob": "$\\theta_{z}$"
}, inplace=True)

print(articles)

articles["Sujet"] = articles["Sujet"].apply(lambda s: "\\\\ ".join(textwrap.wrap(s, width=15)))
articles["Sujet"] = articles["Sujet"].apply(lambda s: '\\begin{tabular}{l}' + s +'\\end{tabular}')

articles["Article"] = articles["Article"].map(lambda s: f"``{s}''")

latex = articles.reset_index(drop=True).set_index(["Sujet", "Article"]).to_latex(
    columns=["Catégories", "$\\theta_{z}$"],
    longtable=True,
    sparsify=True,
    multirow=True,
    multicolumn=True,
    position='H',
    column_format='p{0.25\\textwidth}|p{0.555\\textwidth}|p{0.145\\textwidth}|p{0.05\\textwidth}',
    escape=False,
    caption="\\textbf{Sélection de trois articles emblématiques pour chacun des sujets associés à la supersymétrie}. Les articles sont sélectionnés parmi ceux qui mentionnent la supersymétrie dans leur résumé.",
    label="table:emblematic_articles"
).replace("Continued on next page", "Suite page suivante")

with open("analyses/emblematic_articles.tex", "w+") as fp:
    fp.write(latex)
