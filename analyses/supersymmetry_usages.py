import pandas as pd 
import numpy as np 
import tomotopy as tp 

mdl = tp.CTModel.load("output/hep-ct-75-0.1-0.001-130000-20/model")

topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

p_w_t = topic_term_dists
p_t = np.sum(doc_topic_dists.transpose()*doc_lengths,axis=1)/np.sum(doc_lengths)
p_w = term_frequency/np.sum(doc_lengths)
p_t_w = (p_w_t.T*p_t).T/p_w

data = []
terms = ['supersymmetry', 'supersymmetric', 'susy']

for term in terms:
    w = vocab.index(term)
    topic_word = p_t_w[:,w]
    largest_idx = topic_word.argsort()[-5:][::-1]
    for idx in largest_idx:
        data.append({
            'term': term,
            'topic': idx,
            'p_t_w': topic_word[idx]
        })

pd.DataFrame(data).to_csv('output/supersymmetry_usages.csv')