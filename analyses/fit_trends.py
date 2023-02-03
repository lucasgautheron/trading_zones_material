import pandas as pd 
import numpy as np

import matplotlib
from matplotlib import pyplot as plt 
import seaborn as sns
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

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression

from scipy import stats

import argparse

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Source: https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def is_susy(s: str):
    return "supersymmetr" in s or "susy" in s

parser = argparse.ArgumentParser("extracting correlations")
parser.add_argument("--articles", default="output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet")
parser.add_argument("--since", help="since year", type=int, default=2011)
parser.add_argument("--until", help="until year", type=int, default=2019)
parser.add_argument("--domain", choices=["hep", "susy"], default="susy")
parser.add_argument("--descriptions", default="output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")
args = parser.parse_args()

years = np.arange(args.since, args.until+1)
n_years = len(years)

articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "date_created", "pacs_codes", "categories", "abstract", "title"]]

if args.domain == "susy":
  articles = articles[(articles["abstract"].str.lower().map(is_susy) == True) | (articles["title"].str.lower().map(is_susy) == True)]

articles["article_id"] = articles["article_id"].astype(int)
articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(int)

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

fits = []

for topic in range(n_topics):
    y = cumprobs[:,topic]/counts

    reg = LinearRegression().fit(years.reshape(-1, 1), y)

    r, p, lo_95, hi_95 = pearsonr_ci(years, y)
    r, p, lo_99, hi_99 = pearsonr_ci(years, y, alpha=0.01)

    fits.append({
        'topic': topic,
        'r2': reg.score(years.reshape(-1, 1), y)**2,
        'slope': reg.coef_[0],
        'lower_95': lo_95,
        'high_95': hi_95,
        'lower_99': lo_99,
        'high_99': hi_99
    })

fits = pd.DataFrame(fits)
fits = fits.merge(pd.read_csv(args.descriptions))
fits['95_significant'] = fits['lower_95']*fits['high_95'] > 0
fits['99_significant'] = fits['lower_99']*fits['high_99'] > 0
fits.sort_values("slope", ascending=True, inplace=True)
fits.to_csv('output/fits.csv')

significant_dec = fits[fits["99_significant"]==True].head(3)
significant_inc = fits[fits["99_significant"]==True].tail(3)

fig, axes = plt.subplots(1,2,sharey=True)

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3']

ax = axes[0]
n = 0
for topic in significant_dec.to_dict(orient="records"):
    ax.plot(
        years,
        cumprobs[:,topic['topic']]/counts,
        color = colors[n]
    )
    ax.scatter(
        years,
        cumprobs[:,topic['topic']]/counts,
        label=topic['description'],
        color = colors[n]
    )
    n +=1

ax.set_ylabel("Average relative contribution of each topic per year ($\\bar{\\theta_z}$)")
ax.set_xlim(years.min(), years.max())
ax.legend(fontsize='x-small', loc="upper right")

ax = axes[1]
n = 0
for topic in significant_inc.to_dict(orient="records"):
    ax.plot(
        years,
        cumprobs[:,topic['topic']]/counts,
        color = colors[n]
    )
    ax.scatter(
        years,
        cumprobs[:,topic['topic']]/counts,
        label=topic['description'],
        color = colors[n]
    )
    n +=1

ax.set_xlim(years.min(), years.max())
ax.legend(fontsize='x-small', loc="upper right")

fig.suptitle(
    "Coldest topics (left) and hottest topics (right) – {}, {}-{}".format(
    "high-energy physics" if args.domain == "hep" else "supersymmetry",
    args.since,
    args.until
  )
)
plt.savefig(f"plots/hot_cold_topics_hep_{args.since}_{args.until}_{args.domain}.pgf", bbox_inches="tight")
plt.savefig(f"plots/hot_cold_topics_hep_{args.since}_{args.until}_{args.domain}.pdf", bbox_inches="tight")
plt.savefig(f"plots/hot_cold_topics_hep_{args.since}_{args.until}_{args.domain}.eps", bbox_inches="tight")
