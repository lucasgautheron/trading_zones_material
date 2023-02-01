import nest_asyncio
nest_asyncio.apply()
del nest_asyncio
import stan

import pandas as pd
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import beta

import argparse

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


def is_susy(abstract: str):
    abstract = abstract.lower()

    return (
        "supersymmetry" in abstract
        or "supersymmetric" in abstract
        or "susy" in abstract
    )


def is_hep(categories: str):
    return any(["-HEP" in x for x in categories])

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=["compute", "plot"])
args = parser.parse_args()

if args.action == "compute":
    articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["article_id", "categories", "date_created"]]
    articles["is_hep"] = articles.categories.map(is_hep)
    articles["is_th"] = articles.categories.map(lambda l: "Theory-HEP" in l)
    articles["is_exp"] = articles.categories.map(lambda l: "Experiment-HEP" in l)
    articles["is_pheno"] = articles.categories.map(lambda l: "Phenomenology-HEP" in l)
    #articles["is_susy"] = articles["abstract"].map(is_susy)
    articles["year"] = articles["date_created"].str[:4].replace('', 0).astype(float).fillna(0).astype(int)

    articles = articles[articles["is_hep"]]
    articles = articles[(articles["year"] >= 1980) & (articles["year"] < 2020)]

    authors = pd.read_parquet("projects/inspire-harvest/database/articles_authors.parquet")[["article_id", "bai"]]
    authors = authors.merge(articles[["article_id", "is_th", "is_exp", "is_pheno"]], how='right')

    authors = authors.groupby("bai").agg(
        th = ('is_th', 'sum'),
        exp = ('is_exp', 'sum'),
        ph = ('is_pheno', 'sum'),
        total = ('article_id', 'count'),
        th_frac = ('is_th', 'mean'),
        exp_frac = ('is_exp', 'mean'),
        ph_frac = ('is_pheno', 'mean')
    )

    authors = authors[authors['total'] >= 3]
    authors = authors.sample(frac=1).head(2500)
    authors.to_parquet('output/social_divide_authors.parquet')

    data = {
        'n_authors': len(authors),
        'n_types': 3,
        'n_articles': authors['total'].astype(int).values
    }

    counts_per_type = np.zeros((len(authors),3))
    for i,t in enumerate(['th', 'exp', 'ph']):
        counts_per_type[:,i] = authors[t].astype(int).values

    data['counts_per_type'] = counts_per_type.astype(int)

    stan_code = """
    data {
        int n_authors;
        int n_types;
        int n_articles[n_authors];
        int counts_per_type[n_authors,n_types];
    }

    parameters {
        matrix <lower=0, upper=1> [n_authors, n_types] probs;
        vector<lower=0>[n_types] alphas;
        vector<lower=0>[n_types] betas;

    //    vector<lower=0,upper=1>[n_types] mus;
    //    vector<lower=0.1>[n_types] etas;
    }

    // transformed parameters {
    //     vector<lower=0>[n_types] alphas;
    //     vector<lower=0>[n_types] betas;
    // 
    //     for (i in 1:n_types) {
    //         alphas[i] = mus[i] * etas[i];
    //         betas[i] = (1-mus[i]) * etas[i];
    //     }
    // }

    model {
        for (i in 1:n_authors) {
            for (j in 1:n_types) {
                probs[i,j] ~ beta(alphas[j], betas[j]);
                counts_per_type[i,j] ~ binomial(n_articles[i], probs[i,j]);
            }
        }

        for (i in 1:n_types) {
            alphas[i] ~ exponential(1);
            betas[i] ~ exponential(1);
        }
    }

    """

    posterior = stan.build(stan_code, data = data)
    fit = posterior.sample(num_chains = 4, num_samples = 1000)
    df = fit.to_frame()

    df.to_parquet(f'output/social_divide_samples.parquet')
else:
    df = pd.read_parquet(f'output/social_divide_samples.parquet')
    data = {'n_authors': 2500, 'n_types': 3}

n = len(df)

cats = ['th','exp','ph']
probs = dict()
alphas = dict()
betas = dict()

for t in np.arange(data['n_types']):
    probs[t] = [df[f'probs.{k}.{t+1}'].values for k in 1+np.arange(data['n_authors'])]
    probs[t] = np.array(probs[t])

    alphas[t] = df[f'alphas.{t+1}'].values
    betas[t] = df[f'betas.{t+1}'].values
    
bins = np.linspace(0,1,20,True)

hats = np.zeros((data['n_types'], n, len(bins)-1))
    
for line in np.arange(n):
    for j in np.arange(data['n_types']):
        hats[j,line,:] = np.histogram(
            np.array([df[f'p_hat.{i+1}.{j+1}'].iloc[line] for i in np.arange(data['n_authors'])]),
            bins=bins
        )[0]

fig, axes = plt.subplots(nrows=1,ncols=3,sharey=True)

authors = pd.read_parquet(f'output/social_divide_authors.parquet')

x = np.linspace(0,1,100,True)
for i, t in enumerate([1,2,0]):
    ax = axes[i]

#     ax.hist(
#         probs[t].flatten(),
#         bins=np.linspace(0,1,20,True),
#         histtype="step",
#         color='black',
#         density=True,
#         lw=1,
#         label="$p_{ij}$ (modèle)"
#     )

    m = np.mean(hats[t, :, :]/data['n_authors']/(bins[1]-bins[0]), axis=0)
    
    ax.scatter(
        (bins[:-1]+bins[1:])/2,
        m,
        color="black",
        s=1,
        label="$\hat{p}_{ij}$ (modèle)"

    )

    lower = m-np.quantile(hats[t, :, :]/data['n_authors']/(bins[1]-bins[0]), axis=0, q=0.05/2)
    upper = np.quantile(hats[t, :, :]/data['n_authors']/(bins[1]-bins[0]), axis=0, q=1-0.05/2)-m

    ax.errorbar(
        (bins[:-1]+bins[1:])/2,
        m,
        (lower, upper),
        color="black",
        lw=1,
        ls="None"
    )

    ax.hist(
        authors[f"{cats[t]}_frac"],
        bins=np.linspace(0,1,20,True),
        histtype="step",
        color='blue',
        density=True,
        lw=1,
        linestyle='-',
        label="$\hat{p}_{ij}$ (données)"
    )

    ax.set_yscale('log')
    # for i in np.arange(100):
    #     ax.plot(x, beta.pdf(x, alphas[t][-i], betas[t][-i]), alpha=0.01, color='blue')

    ax.set_xlim(0,1)

    cat = cats[t]
    ax.set_xlabel(f'$p_{{i,\\mathrm{{{cat}}}}}$')

plt.legend()

actors_desc = "physiciens" if args.target == "authors" else "institutions"
fig.suptitle("Part des publications des "+actors_desc+" appartenant aux catégories\n``Expérience'' ($p_{exp}$), ``Phénoménologie'' ($p_{ph}$), ``Théorie'' ($p_{th}$)")
plt.savefig(f"plots/social_divide_gof_{args.target}.pgf")
plt.savefig(f"plots/social_divide_gof_{args.target}.pdf")

