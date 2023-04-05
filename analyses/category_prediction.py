from AbstractSemantics.terms import TermExtractor
import pandas as pd
import numpy as np
from os.path import join as opj
from os.path import exists

import itertools
from functools import partial
from collections import defaultdict

import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

import multiprocessing as mp

from matplotlib import pyplot as plt

import argparse
import yaml
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CT Model')
    parser.add_argument('location', help='model directory')
    parser.add_argument('filter', choices=['categories', 'keywords', 'no-filter'], help='filter type')
    parser.add_argument('--values', nargs='+', default=[], help='filter allowed values')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--constant-sampling', type=int, default=0)
    parser.add_argument('--reuse-articles', default=False, action="store_true", help="reuse article selection")
    parser.add_argument('--nouns', default=False, action="store_true", help="include nouns")
    parser.add_argument('--adjectives', default=False, action="store_true", help="include adjectives")
    parser.add_argument('--lemmatize', default=False, action="store_true", help="stemmer")
    parser.add_argument('--lemmatize-ngrams', default=False, action="store_true", help="stemmer")
    parser.add_argument('--remove-latex', default=False, action="store_true", help="remove latex")
    parser.add_argument('--limit-redundancy', default=False, action="store_true", help="limit redundancy")
    parser.add_argument('--add-title', default=False, action="store_true", help="include title")
    parser.add_argument('--top-unithood', type=int, default=20000, help='top unithood filter')
    parser.add_argument('--min-token-length', type=int, default=0, help='minimum token length')
    parser.add_argument('--min-df', type=int, default=0, help='min_df')
    parser.add_argument('--reuse-stored-vocabulary', default=False, action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(["output/category_prediction", "categories", "--values", "Experiment-HEP", "Phenomenology-HEP", "Theory-HEP", "--samples", "110000", "--nouns", "--lemmatize", "--lemmatize-ngrams", "--remove-latex", "--add-title", "--top-unithood", "1000", "--threads", "16"])

    with open(opj(args.location, "params.yml"), "w+") as fp:
        yaml.dump(args, fp)

    articles = pd.read_parquet("inspire-harvest/database/articles.parquet")[["title", "abstract", "article_id", "date_created", "categories"]]

    if args.add_title:
        articles["abstract"] = articles["abstract"].str.cat(articles["title"])

    articles.drop(columns = ["title"], inplace=True)

    if args.remove_latex:
        articles['abstract'] = articles['abstract'].apply(lambda s: re.sub('$[^>]+$', '', s))

    articles = articles[articles["abstract"].map(len)>=100]
    articles["abstract"] = articles["abstract"].str.lower()

    articles = articles[articles["date_created"].str.len() >= 4]
    articles["year"] = articles["date_created"].str[:4].astype(int)-1980
    articles = articles[(articles["year"] >= 0) & (articles["year"] <= 40)]
    articles["year_group"] = articles["year"]//5

    if args.reuse_articles:
        used = pd.read_csv(opj(args.location, 'articles.csv'))
        articles = articles[articles["article_id"].isin(used["article_id"])]
    else:
        articles = articles[~articles["abstract"].isnull()]

        if args.constant_sampling > 0:
            articles = articles.groupby("year").head(args.constant_sampling)

        keep = pd.Series([False]*len(articles), index=articles.index)

        print("Applying filter...")
        if args.filter == 'keywords':
            for value in args.values:
                keep |= articles["abstract"].str.contains(value)
        elif args.filter == 'categories':
            for value in args.values:
                keep |= articles["categories"].apply(lambda l: value in l)

        articles = articles[keep==True]
        articles = articles.sample(frac=1).head(args.samples)
        articles[["article_id"]].to_csv(opj(args.location, 'articles.csv'))

    articles.reset_index(inplace = True)

    print("Extracting n-grams...")
    extractor = TermExtractor(articles["abstract"].tolist(), limit_redundancy=args.limit_redundancy)

    if args.nouns:
        extractor.add_patterns([["NN.*"]])

    if args.adjectives:
        extractor.add_patterns([["^JJ$"]])

    ngrams = extractor.ngrams(threads=args.threads,lemmatize=args.lemmatize,lemmatize_ngrams=args.lemmatize_ngrams)
    ngrams = map(lambda l: [" ".join(n) for n in l], ngrams)
    ngrams = list(ngrams)

    articles["ngrams"] = ngrams

    print("Deriving vocabulary...")
    if not args.reuse_stored_vocabulary:
        ngrams_occurrences = defaultdict(int)

        for ngrams in articles["ngrams"].tolist():
            _ngrams = set(ngrams)
            for ngram in _ngrams:
                ngrams_occurrences[ngram] += 1

        ngrams_occurrences = pd.DataFrame(
            {"ngram": ngrams_occurrences.keys(), "count": ngrams_occurrences.values()}
        )
        ngrams_occurrences["unithood"] = (
            np.log(2 + ngrams_occurrences["ngram"].str.count(" "))
            * ngrams_occurrences["count"]
        )
        ngrams_occurrences["unithood"] /= len(articles)
        ngrams_occurrences.set_index("ngram", inplace=True)

        ngrams_occurrences["len"] = ngrams_occurrences.index.map(len)
        ngrams_occurrences = ngrams_occurrences[ngrams_occurrences["len"] > 1]

        top = ngrams_occurrences.sort_values("unithood", ascending=False).head(
            args.top_unithood
        )

        top.to_csv(opj(args.location, "ngrams.csv"))

    
    selected_ngrams = pd.read_csv(opj(args.location, 'ngrams.csv'))['ngram'].tolist()

    vocabulary = {
        n: i
        for i, n in enumerate(selected_ngrams)
    }

    ngrams = articles["ngrams"].tolist()
    ngrams = [[ngram for ngram in _ngrams if ngram in selected_ngrams] for _ngrams in ngrams]

    bow = [[vocabulary[ngram] for ngram in _ngrams] for _ngrams in ngrams]
    bow = [[_ngrams.count(i) for i in range(len(selected_ngrams))] for _ngrams in bow]
    bow = np.array(bow)

    tfidf = TfidfTransformer()
    bow_tfidf = tfidf.fit_transform(bow).todense().tolist()
    articles["bow_tfidf"] = bow_tfidf

    cat_classifier = MultiLabelBinarizer(sparse_output=False)
    articles["categories"] = articles["categories"].map(lambda l: list(set(l)&{"Experiment-HEP", "Phenomenology-HEP", "Theory-HEP"}))
    cats = cat_classifier.fit_transform(articles["categories"]).tolist()
    articles["cats"] = cats

    training, validation = train_test_split(articles, train_size=100000/110000)

    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score

    dummies = dict()
    fit = dict()
    scores = dict()
    f1 = dict()

    dummies_scores = dict()
    dummies_f1 = dict()

    score_vs_vocab_size = []

    for vocab in [50] + list(np.arange(125, 1000, 125)):
        score = 0
        for i in range(3):
            dummies[i] = DummyClassifier(strategy="most_frequent")
            dummies[i].fit(np.stack(training["bow_tfidf"].values), np.stack(training["cats"].values).astype(int)[:,i])

            fit[i] = LogisticRegression(random_state=0, max_iter=200).fit(np.stack(training["bow_tfidf"].values)[:,0:vocab], np.stack(training["cats"].values).astype(int)[:,i])
            y_hat = np.stack(validation["cats"].values).astype(int)[:,i]
            scores[i] = fit[i].score(np.stack(validation["bow_tfidf"].values)[:,0:vocab], y_hat)
            f1[i] = f1_score(y_hat, fit[i].predict(np.stack(validation["bow_tfidf"].values)[:,0:vocab]))
            score += f1[i]

            dummies_scores[i] = dummies[i].score(np.stack(validation["bow_tfidf"].values), y_hat)
            dummies_f1[i] = f1_score(y_hat, dummies[i].predict(np.stack(validation["bow_tfidf"].values)))

        print(vocab, score)
        score_vs_vocab_size.append({
            'vocab': vocab,
            'f1': score,
            "acc_0": scores[0],
            "acc_1": scores[1],
            "acc_2": scores[2],
            "baseline_acc_0": dummies_scores[0],
            "baseline_acc_1": dummies_scores[1],
            "baseline_acc_2": dummies_scores[2],
            "f1_0": f1[0],
            "f1_1": f1[1],
            "f1_2": f1[2],
        })

    score_vs_vocab_size = pd.DataFrame(score_vs_vocab_size)
    score_vs_vocab_size.to_csv(opj(args.location, "vocab_performance.csv"))

    vocab = 500

    scores = dict()
    f1 = dict()

    dummies_scores = dict()
    dummies_f1 = dict()

    frequency = np.stack(articles["cats"].values).sum(axis=0)
    results = []

    inv_vocabulary = {
        vocabulary[v]: v
        for v in vocabulary
    }

    for i in range(3):
        dummies[i] = DummyClassifier(strategy="most_frequent")
        dummies[i].fit(np.stack(training["bow_tfidf"].values), np.stack(training["cats"].values).astype(int)[:,i])
        
        fit[i] = LogisticRegression(random_state=0,max_iter=200).fit(np.stack(training["bow_tfidf"].values)[:,0:vocab], np.stack(training["cats"].values).astype(int)[:,i])

        y_hat = np.stack(validation["cats"].values).astype(int)[:,i]
        scores[i] = fit[i].score(np.stack(validation["bow_tfidf"].values)[:,0:vocab], y_hat)
        f1[i] = f1_score(y_hat, fit[i].predict(np.stack(validation["bow_tfidf"].values)[:,0:vocab]))
        
        dummies_scores[i] = dummies[i].score(np.stack(validation["bow_tfidf"].values), y_hat)
        dummies_f1[i] = f1_score(y_hat, dummies[i].predict(np.stack(validation["bow_tfidf"].values)))

        for j in range(vocab):
            results.append({
                'term': inv_vocabulary[j],
                'category': cat_classifier.inverse_transform(np.array([np.identity(3)[i,:]]))[0][0],
                'coef': fit[i].coef_[0,j],
                'rank': j
            })

        predictions = fit[i].predict(np.stack(validation["bow_tfidf"].values)[:,0:vocab])
        dummy_predictions = dummies[i].predict(np.stack(validation["bow_tfidf"].values)[:,0:vocab])

        validation[f"accurate_{i}"] = predictions==y_hat
        validation[f"dummy_accurate_{i}"] = dummy_predictions==y_hat
        validation[f"truth_{i}"] = y_hat

    validation.groupby("year_group").agg(
        accurate_0=("accurate_0", "mean"),
        accurate_1=("accurate_1", "mean"),
        accurate_2=("accurate_2", "mean"),
        dummy_accurate_0=("dummy_accurate_0", "mean"),
        dummy_accurate_1=("dummy_accurate_1", "mean"),
        dummy_accurate_2=("dummy_accurate_2", "mean"),
        truth_0=("truth_0", "sum"),
        truth_1=("truth_1", "sum"),
        truth_2=("truth_2", "sum"),
        count_0=("truth_0", "count"),
        count_1=("truth_1", "count"),
        count_2=("truth_2", "count"),
    ).to_csv(opj(args.location, "accuracy_per_period.csv"))

    kfold = []

    
    for year_group, test in articles.groupby("year_group"):
        train = articles[articles["year_group"] != year_group]
        accurate = np.zeros(3)
        dummy_accurate = np.zeros(3)
        truth = np.zeros(3)
        count = np.zeros(3)

        for i in range(3):
            kfold_fit = LogisticRegression(random_state=0,max_iter=200).fit(np.stack(train["bow_tfidf"].values)[:,0:vocab], np.stack(train["cats"].values).astype(int)[:,i])
            y_hat = np.stack(test["cats"].values).astype(int)[:,i]

            predictions = kfold_fit.predict(np.stack(test["bow_tfidf"].values)[:,0:vocab])
            dummy_predictions = dummies[i].predict(np.stack(test["bow_tfidf"].values)[:,0:vocab])

            accurate[i] = (predictions==y_hat).mean()
            dummy_accurate[i] = (dummy_predictions==y_hat).mean()
            truth[i] = y_hat.mean()
            count[i] = len(test)

        kfold.append({
            "year_group": year_group,
            "accurate_0": accurate[0],
            "accurate_1": accurate[1],
            "accurate_2": accurate[2],
            "dummy_accurate_0": dummy_accurate[0],
            "dummy_accurate_1": dummy_accurate[1],
            "dummy_accurate_2": dummy_accurate[2],
            "truth_0": truth[0],
            "truth_1": truth[1],
            "truth_2": truth[2],
            "count_0": count[0],
            "count_1": count[1],
            "count_2": count[2],
        })

    pd.DataFrame(kfold).to_csv(opj(args.location, "accuracy_per_period_kfold.csv"))

    results = pd.DataFrame(results)
    results["drop"] = False

    bow = (bow>=1).astype(int)
    num = np.outer(bow[:3000,:vocab].sum(axis=0),bow[:3000,:vocab].sum(axis=0))/(3000**2)
    den = np.tensordot(bow[:3000,:vocab], bow[:3000,:vocab], axes=([0],[0]))/3000
    npmi = np.log(num)/np.log(den)-1

    x, y = np.where(npmi-np.identity(vocab)>=0.95)
    for k,_ in enumerate(x):
        i = x[k]
        j = y[k]

        a = inv_vocabulary[i]
        b = inv_vocabulary[j]

        if (not (a in b or b in a)):
            continue

        if i > j:
            results.loc[results['rank'] == i, 'drop'] = True
        else:
            results.loc[results['rank'] == j, 'drop'] = True

    results = results[results["drop"]==False]
    results = results[results["term"].str.match("^[a-zA-Z--- ]*$")]

    results = results.pivot(index="term",columns="category",values="coef")
    results["ph_minus_th"] = results["Phenomenology-HEP"]-results["Theory-HEP"]
    results["ph_minus_exp"] = results["Phenomenology-HEP"]-results["Experiment-HEP"]
    results.sort_values("ph_minus_th").to_csv(opj(args.location, "results.csv"))

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

    cats = {"exp": "Experiment-HEP", "th": "Theory-HEP"}
    cats_friendly = {"th": "Theory", "exp": "Experiment"}

    table = []

    for cat in cats:
        top = results[results["Phenomenology-HEP"]>0].sort_values(f"ph_minus_{cat}", ascending=False).head(40).index.values
        bottom = results[results[cats[cat]]>0].sort_values(f"ph_minus_{cat}", ascending=True).head(40).index.values
        
        table.append({
            'Reference category': cats_friendly[cat],
            'Relation to phenomenology': "Vocabulary specific to phenomenology",
            'words': ", ".join(top)
        })
        
        table.append({
            'Reference category': cats_friendly[cat],
            'Relation to phenomenology': "Vocabulary specific to theory",
            'words': ", ".join(bottom)
        })
        
    table = pd.DataFrame(table)
    table = table.pivot(index="Reference category", columns="Relation to phenomenology", values="words")

    with pd.option_context("display.max_colwidth", None):
        latex = table.to_latex(
            longtable=True,
            multirow=True,
            multicolumn=True,
            bold_rows=True,
            header=True,
            index_names=False,
            column_format='p{3cm}|p{5cm}|p{5cm}',
            caption="Vocabulary specific to each category. The left column lists expressions that discriminate experiment and theory from phenomenology. The right column lists expressions that are the most specific to phenomenology and foreigh to experiment and theory.",
            label="table:specific_pheno_vocabulary"
        )

    with open("tables/specific_vocabulary.tex", "w+") as fp:
        fp.write(latex)


    for cat in ["th", "exp"]:
        table = []
        top = results[results["Phenomenology-HEP"]>0].sort_values(f"ph_minus_{cat}", ascending=False).head(45).index.values
        bottom = results[results[cats[cat]]>0].sort_values(f"ph_minus_{cat}", ascending=True).head(45).index.values
            
        table.append({
            'Reference category': cats_friendly[cat],
            'Relation to phenomenology': "Vocabulary specific to phenomenology",
            'words': ", ".join(top)
        })
        
        table.append({
            'Reference category': cats_friendly[cat],
            'Relation to phenomenology': f"Vocabulary specific to {cats_friendly[cat].lower()}",
            'words': ", ".join(bottom)
        })
            
        table = pd.DataFrame(table)
        table = table.pivot(index="Reference category", columns="Relation to phenomenology", values="words")

        caption = f"Vocabulary specific to phenomenology (left column) versus {cats_friendly[cat].lower()} (right column)."

        with pd.option_context("display.max_colwidth", None):
            latex = table.to_latex(
                longtable=True,
                multirow=True,
                multicolumn=True,
                bold_rows=True,
                header=True,
                index_names=False,
                index=False,
                column_format='p{7cm}|p{7cm}',
                caption=caption,
                label=f"table:specific_pheno_vocabulary_{cat}_ph",
                columns = ["Vocabulary specific to phenomenology", f"Vocabulary specific to {cats_friendly[cat].lower()}"]
            )

        with open(f"tables/specific_vocabulary_{cat}_ph.tex", "w+") as fp:
            fp.write(latex)
