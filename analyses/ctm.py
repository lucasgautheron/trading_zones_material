
from AbstractSemantics.terms import TermExtractor
import pandas as pd
import numpy as np
from os.path import join as opj
from os.path import exists

import itertools
from functools import partial
from collections import defaultdict

import re

import tomotopy as tp

from sklearn.model_selection import train_test_split

import tqdm

import multiprocessing as mp

from matplotlib import pyplot as plt

import argparse
import yaml
import sys

parser = argparse.ArgumentParser('CT Model')
parser.add_argument('location', help='model directory')
parser.add_argument('filter', choices=['categories', 'keywords', 'no-filter'], help='filter type')
parser.add_argument('--values', nargs='+', default=[], help='filter allowed values')
parser.add_argument('--samples', type=int, default=100000)
parser.add_argument('--constant-sampling', type=int, default=0)
parser.add_argument('--reuse-articles', default=False, action="store_true", help="reuse article selection")
parser.add_argument('--nouns', default=False, action="store_true", help="include nouns")
parser.add_argument('--adjectives', default=False, action="store_true", help="include adjectives")
parser.add_argument('--lemmatize', default=False, action="store_true", help="stemmer")
parser.add_argument('--remove-latex', default=False, action="store_true", help="remove latex")
parser.add_argument('--limit-redundancy', default=False, action="store_true", help="limit redundancy")
parser.add_argument('--add-title', default=False, action="store_true", help="include title")
parser.add_argument('--top-unithood', type=int, default=20000, help='top unithood filter')
parser.add_argument('--min-token-length', type=int, default=0, help='minimum token length')
parser.add_argument('--min-df', type=int, default=0, help='min_df')
# parser.add_argument('--top-termhood', type=int, default=15000, help='top termhood filter')
parser.add_argument('--reload-model', default=False, action="store_true", help="reload saved model")
parser.add_argument('--reuse-stored-vocabulary', default=False, action='store_true')
parser.add_argument('--compute-best-params', action='store_true', help='optimize hyperparameters (maximzing C_v)', required=False)
parser.add_argument('--reuse-best-params', action='store_true', help='re-use optimal hyperparameters', required=False)
parser.add_argument('--topics', type=int, default=8, help='topics')
parser.add_argument('--alpha', default=0.1, type=float, help='LDA alpha prior')
parser.add_argument('--eta', default=0.01, type=float, help='LDA beta(eta) prior')
parser.add_argument('--threads', type=int, default=4)
args = parser.parse_args()

if __name__ == "__main__":

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

    ngrams = extractor.ngrams(threads=args.threads,lemmatize=args.lemmatize)
    ngrams = map(lambda l: [" ".join(n) for n in l], ngrams)
    ngrams = list(ngrams)

    articles["ngrams"] = ngrams

    print("Deriving vocabulary...")
    if not args.reuse_stored_vocabulary:
        ngrams_occurrences = defaultdict(int)
        ngrams_cooccurrences = defaultdict(int)

        termhood = defaultdict(int)

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

        top_unithood = ngrams_occurrences.sort_values("unithood", ascending=False).head(
            args.top_unithood
        )
        top = top_unithood
        
        top.to_csv(opj(args.location, "ngrams.csv"))
    

    selected_ngrams = set(pd.read_csv(opj(args.location, 'ngrams.csv'))['ngram'].tolist())

    ngrams = articles["ngrams"].tolist()
    ngrams = [[ngram for ngram in _ngrams if ngram in selected_ngrams] for _ngrams in ngrams]

    training_ngrams, validation_ngrams = train_test_split(ngrams, train_size=0.9)

    print("Creating tomotopy copora...")
    training_corpus = tp.utils.Corpus()
    for doc in training_ngrams:
        training_corpus.add_doc(words=doc)

    validation_corpus = tp.utils.Corpus()
    for doc in validation_ngrams:
        validation_corpus.add_doc(words=doc)

    if args.compute_best_params:
        topics = list(range(25, 100, 25)) + list(range(100, 200, 50))
        alphas = np.logspace(-2, 0, 3, True)
        etas = np.logspace(-3, -1, 3, True)

        model_results = {
            'topics': [],
            'alphas': [],
            'etas': [],
            'u_mass': [],
            'c_uci': [],
            'c_npmi': [],
            'c_v': [],
            'train_ll_per_word': [],
            'validation_ll': [],
            'documents': [],
            'words': [],
            'perplexity': [],
            'train_perplexity': []
        }

        try:
            done = pd.read_csv(opj(args.location, 'lda_tuning_results.csv'))
            model_results = done.to_dict(orient="list")
            print(model_results)
        except Exception as e:
            print(e)
            done = None

        with tqdm.tqdm(total=len(topics)*len(alphas)*len(etas)) as pbar:
            for k in topics:
                for alpha in alphas:
                    # alpha = alpha*10/k
                    for eta in etas:

                        print(k, alpha, eta)

                        is_done = done is not None and len(done[(done["topics"] == k) & (done["alphas"] == alpha) & (done["etas"] == eta)]) > 0
                        if is_done:
                            print("already done")
                            continue

                        try:
                            mdl = tp.CTModel(
                                tw=tp.TermWeight.ONE,
                                corpus=training_corpus,
                                k=k,
                                min_df=3,
                                smoothing_alpha=alpha,
                                eta=eta
                            )
                            mdl.train(0)

                            prev_ll_per_word = None

                            for _ in range(0, 100, 10):
                                mdl.train(10)
                                print('Iteration: {:05}\tll per word: {:.5f}'.format(mdl.global_step, mdl.ll_per_word))

                                if prev_ll_per_word is not None and prev_ll_per_word > mdl.ll_per_word:
                                    print("stopping here")
                                    break
                                else:
                                    prev_ll_per_word = mdl.ll_per_word

                        except:
                            print("failed")
                            pbar.update(1)
                            continue

                        for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
                            coh = tp.coherence.Coherence(mdl, coherence=preset)
                            average_coherence = coh.get_score()
                            model_results[preset].append(average_coherence)

                        res, total_ll = mdl.infer(validation_corpus, together=True)

                        _ll = np.array([doc.get_ll() for doc in res])
                        words = np.array([len(doc.words) for doc in res])
                        
                        perplexity = np.exp(-np.sum(total_ll)/np.sum(words))
                        print(perplexity, mdl.perplexity)
                        print(-np.sum(total_ll)/np.sum(words), np.log(mdl.perplexity), -np.sum(total_ll)/np.sum(words)/np.log(mdl.perplexity))

                        #print(total_ll, _ll)

                        print(f"Topics: {k}, Perplexity: {perplexity}")
                        print(mdl.ll_per_word)
                        print(mdl.perplexity)
                        print(mdl.num_words)
                        
                        model_results['train_ll_per_word'].append(mdl.ll_per_word)
                        model_results['validation_ll'].append(np.sum(total_ll))
                        model_results['documents'].append(len(res))
                        model_results['words'].append(np.sum(words))
                        model_results['perplexity'].append(perplexity)
                        model_results['train_perplexity'].append(mdl.perplexity)
                        model_results['topics'].append(k)
                        model_results['alphas'].append(alpha)
                        model_results['etas'].append(eta)

                        pd.DataFrame(model_results).to_csv(opj(args.location, 'lda_tuning_results.csv'), index=False)

                        pbar.update(1)

    params = {'topics': args.topics}

    if not args.reload_model:
        print("Training LDA...")
        min_df = args.min_df
        print(min_df)

        mdl = tp.CTModel(
            tw=tp.TermWeight.ONE,
            corpus=training_corpus,
            k=params['topics'],
            min_df=min_df,
            smoothing_alpha=args.alpha,
            eta=args.eta
        )
        mdl.train(0)

        print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
        print('Removed top words:', mdl.removed_top_words)
        print('Training...', file=sys.stderr, flush=True)
        for _ in range(0, 250, 10):
            mdl.train(10)
            print('Iteration: {:05}\tll per word: {:.5f}'.format(mdl.global_step, mdl.ll_per_word))

        import pyLDAvis

        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency,
            start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
            sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
        )
        pyLDAvis.save_html(prepared_data, opj(args.location, 'ldavis.html'))

        print('Saving...', file=sys.stderr, flush=True)
        mdl.save(opj(args.location, "model"), True)
    else:
        print("Loading pre-trained model...")
        mdl = tp.CTModel.load(opj(args.location, "model"))

    mdl.summary()

    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()

    for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
        coh = tp.coherence.Coherence(mdl, coherence=preset)
        average_coherence = coh.get_score()
        coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
        print('==== Coherence: {} ===='.format(preset))
        print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
        print()

    print("Applying model...")

    used_vocab = set(mdl.used_vocabs)

    articles["ngrams"] = ngrams
    articles = articles[articles["ngrams"].map(len) > 0]
    articles = articles[articles["ngrams"].map(lambda l: len(set(l)&used_vocab) > 0) == True]
    ngrams = articles["ngrams"].tolist()

    corpus = tp.utils.Corpus()
    for doc in ngrams:
        corpus.add_doc(words=doc)

    test_result_cps, ll = mdl.infer(corpus)
    topic_dist = []
    for i, doc in enumerate(test_result_cps):
        print(i, doc)
        dist = doc.get_topic_dist()
        topic_dist.append(dist)

    n = 0
    while exists(opj(args.location, f"topics_{n}.parquet")):
        n +=1 
    
    path = opj(args.location, f"topics_{n}.parquet")

    articles["probs"] = topic_dist
    articles["topics"] = articles["probs"].map(lambda l: ",".join(list(map('{:.6f}'.format, l))))
    articles[["year", "article_id", "topics", "probs"]].to_parquet(path, index=False)

    try:
        descriptions = pd.read_csv(opj(args.location, "descriptions.csv")).set_index("topic")
    except:
        descriptions = None        

    cumprobs = np.zeros((42, mdl.k))
    counts = np.zeros(42)

    for year, _articles in articles.groupby("year"):
        print(year)
        for article in _articles.to_dict(orient = 'records'):
            for topic, prob in enumerate(article['probs']):
                cumprobs[year,topic] += prob

        counts[year] = len(_articles)

    cumprobs.dump(opj(args.location, 'cumsprobs.npy'))
    counts.dump(opj(args.location, 'counts.npy'))

    lines = ['-', '--', '-.', ':', 'dotted', (0, (1, 10)), (0, (3, 10, 1, 10)), (0, (5, 10)), (0, (3, 1, 1, 1, 1, 1)), '-', '--']
    for topic in range(mdl.k):
        plt.plot(
            1980+np.arange(42),
            cumprobs[:,topic],
            linestyle=lines[topic//7],
            label=topic if descriptions is None else descriptions.loc[topic,"description"]
        )

    plt.title("Absolute magnitude of supersymmetry research topics")
    plt.ylabel("Estimated amount of articles\n($\\sum_{d_i \\in \\mathrm{year}} p(t|d_i)$)")

    plt.xlim(1980, 2018)
    plt.legend(fontsize='x-small')

    plt.savefig(opj(args.location, "topics_count.pdf"))
    plt.clf()

    for topic in range(mdl.k):
        plt.plot(
            1980+np.arange(42),
            cumprobs[:,topic]/counts,
            linestyle=lines[topic//7],
            label=topic if descriptions is None else descriptions.loc[topic,"description"]
        )

    plt.title("Relative magnitude of supersymmetry research topics")
    plt.ylabel("Probability of each topic throughout years\n($p(t|\\mathrm{year}$)")

    plt.xlim(1980, 2018)
    plt.legend(fontsize='x-small')

    plt.savefig(opj(args.location, "topics_probs.pdf"))
    plt.clf()
