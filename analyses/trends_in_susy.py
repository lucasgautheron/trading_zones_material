import pandas as pd 
import numpy as np 

from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer

import argparse

selected_topics = [11,33,37,42,60,64]
years = np.arange(1980,2020)

n_years = len(years)

def is_susy(s: str):
    return "supersymmetr" in s or "susy" in s

parser = argparse.ArgumentParser("extracting correlations")
parser.add_argument("articles")
args = parser.parse_args()

articles = pd.read_parquet("../inspire-harvest/database/articles.parquet")[["article_id", "date_created", "pacs_codes", "categories", "abstract"]]
articles = articles[articles["abstract"].str.lower().map(is_susy) == True]
articles["article_id"] = articles["article_id"].astype(int)
articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(float).fillna(0).astype(int)

articles = articles[(articles["year"] >= years.min()) & (articles["year"] <= years.max())]

topics = pd.read_parquet(args.articles)
topics["article_id"] = topics["article_id"].astype(int)
topics["topics"] = topics["probs"]
topics.drop(columns = ["year"], inplace = True)
topics = topics.merge(articles, how="inner", left_on = "article_id", right_on = "article_id")

n_topics = len(topics["topics"].iloc[0])

cumprobs = np.zeros((n_years, n_topics))
counts = np.zeros(n_years)

for year, _articles in topics.groupby("year"):
    for article in _articles.to_dict(orient = 'records'):
        for topic, prob in enumerate(article['probs']):
            cumprobs[year-years.min(),topic] += prob

    counts[year-years.min()] = len(_articles)

for topic in selected_topics:
    plt.plot(
        years,
        cumprobs[:,topic]/counts,
        # linestyle=lines[topic//7],
        label=topic
    )

plt.title("Relative magnitude of topics within abstracts mentioning supersymmetry")
plt.ylabel("Probability of each topic throughout years\n($p(t|\\mathrm{year}$)")

plt.xlim(1980, 2018)
plt.legend(fontsize='x-small')
plt.show()