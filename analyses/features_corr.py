import pandas as pd 
import numpy as np 

from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer

import argparse

parser = argparse.ArgumentParser("extracting correlations")
parser.add_argument('type', choices=["categories", "pacs_codes", "susy"])
parser.add_argument("cond", choices=["cat_topic", "topic_cat", "pearson", "pmi", "npmi"])
parser.add_argument("articles")
parser.add_argument("destination")
parser.add_argument("--descriptions", required=False)
parser.add_argument("--filter", nargs='+', default=[])

args = parser.parse_args([
    "pacs_codes",
    "pmi",
    "output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet",
    "output/pmi.csv",
    "--descriptions", "output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv"
])

def is_susy(s: str):
    return "supersymmetr" in s or "susy" in s


articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "pacs_codes", "categories"] + (["abstract", "title"] if args.type == "susy" else [])]
articles["article_id"] = articles["article_id"].astype(int)

if args.type == "susy":
    articles["susy"] = articles["title"].str.lower().map(is_susy) | articles["abstract"].str.lower().map(is_susy)
    articles["susy"] = articles["susy"].map(lambda x: ["susy"] if x else ["not_susy"])

topics = pd.read_parquet(args.articles)
topics["article_id"] = topics["article_id"].astype(int)
topics["topics"] = topics["probs"]

if 'categories' in topics.columns:
    topics.drop(columns = ['categories'], inplace = True)

topics = topics.merge(articles, how="inner", left_on = "article_id", right_on = "article_id")
topics = topics[topics[args.type].map(len) > 0]

if args.type == "pacs_codes":
    codes = set(pd.read_csv("inspire-harvest/database/pacs_codes.csv")["code"])
    topics = topics[topics["pacs_codes"].map(lambda l: set(l)&codes).map(len) > 0]

X = np.stack(topics["topics"].values)

binarizer = MultiLabelBinarizer()
Y = binarizer.fit_transform(topics[args.type])

n_articles = len(X)
n_topics = X.shape[1]
n_categories = Y.shape[1]

sums = np.zeros((n_topics, n_categories))
topic_probs = np.zeros(n_topics)
p_topic_cat = np.zeros((n_topics, n_categories))
p_cat_topic = np.zeros((n_topics, n_categories))
pearson = np.zeros((n_topics, n_categories))
pmi = np.zeros((n_topics, n_categories))
npmi = np.zeros((n_topics, n_categories))

if args.cond == "pearson":
    for k in range(n_topics):
        for c in range(n_categories):
            pearson[k,c] = np.corrcoef(X[:,k],Y[:,c])[0,1]

for i in range(n_articles):
    for k in range(n_topics):
        sums[k,:] += Y[i,:]*X[i,k]

topic_probs = np.mean(X,axis=0)
cat_probs = np.mean(Y,axis=0)
cat_counts = np.sum(Y,axis=0)

significant_cats = cat_counts>=100

for k in range(n_topics):
    p_cat_topic[k,:] = sums[k,:]/(topic_probs[k]*n_articles)

for c in range(n_categories):
    p_topic_cat[:,c] = sums[:,c]/(cat_probs[c]*n_articles)

for k in range(n_topics):
    pmi[k,:] = np.log(sums[k,:]/(topic_probs[k]*np.sum(Y,axis=0)))

for k in range(n_topics):
    npmi[k,:] = -np.log(sums[k,:]/(topic_probs[k]*np.sum(Y,axis=0)))/np.log(sums[k,:]/n_articles)

cat_classes = np.array([np.identity(n_categories)[cl] for cl in range(n_categories)])
cat_labels = binarizer.inverse_transform(cat_classes)

data = dict()

for c in range(n_categories):
    data[cat_labels[c][0]] = p_topic_cat[:,c] if args.cond == "topic_cat" else (p_cat_topic[:,c] if args.cond == "cat_topic" else (pearson[:,c] if args.cond == "pearson" else (pmi[:,c] if args.cond == "pmi" else npmi[:,c])))

data = pd.DataFrame(data)

if len(args.filter):
    data = data[args.filter]
else:
    cats = map(lambda c: cat_labels[c][0], np.arange(n_categories)[significant_cats])
    data = data[cats]

data["topic"] = data.index

if args.descriptions:
    descriptions = pd.read_csv(args.descriptions)[["topic", "description"]].rename(columns={'description_fr': 'description'})
    data = data.merge(descriptions, how='left', left_index=True,right_on="topic")

data.to_csv(args.destination)

if len(args.filter):
    sns.heatmap(data[args.filter], annot=True, fmt=".2f")
    plt.show()
