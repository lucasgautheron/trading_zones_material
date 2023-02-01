import pandas as pd 
import numpy as np 
import tomotopy as tp 
from os.path import exists
import textwrap

latex_chars = "^+=_"

if not exists("output/top_words.csv"):
    mdl = tp.CTModel.load("output/hep-ct-75-0.1-0.001-130000-20/model")
    top_words = []
    for topic in range(mdl.k):
        words = mdl.get_topic_words(topic, 100)
        words = [
            {
                'topic': topic,
                'word': word,
                'unithood': np.log(2+word.count(' ')),
                'p': p
            }
            for word, p in words
        ]

        top_words += words

    top_words = pd.DataFrame(top_words)
    top_words = top_words[~top_words["word"].str.contains("\\", regex=False)]
    top_words["word"] = top_words["word"].apply(
        lambda w: (
            f"${w}$" if any([c in w for c in latex_chars]) else w
        )
    )
    top_words["word"] = top_words["word"].apply(
        lambda w: (
            w[:-2] + '$' if w[-2:] == '_$' or w[-2:] == '^$' else w
        )
    )
    top_words['x'] = top_words['p']*top_words['unithood']

    top_words = top_words.sort_values(["topic", "x"], ascending=[True, False]).groupby("topic").head(15)
    top_words.to_csv("output/top_words.csv")
else:
    top_words = pd.read_csv("output/top_words.csv")

top_words = top_words.groupby("topic").agg(
    word = ('word', lambda x: ", ".join(x.tolist()))
).reset_index()

top_words = top_words.merge(pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")[["topic", "description"]])

top_words.rename(columns = {
    'word': 'Most frequent expressions',
    "description": "Topic (context)"
}, inplace = True)
top_words.sort_values("Topic (context)", inplace=True)

# top_words["Sujet"] = top_words["Sujet"].apply(lambda s: "\\\\ ".join(textwrap.wrap(s, width=15)))

pd.set_option('display.max_colwidth', None)

latex = top_words.reset_index()[["Topic (context)", "Most frequent expressions"]].set_index(["Topic (context)"]).to_latex(
    longtable=True,
    sparsify=True,
    multirow=True,
    multicolumn=True,
    position='H',
    column_format='p{0.2\\textwidth}|p{0.8\\textwidth}',
    escape=False,
    caption="Most frequent terms for each topic.",
    label="table:top_words"
)

latex = latex.replace('\\\\\n', '\\\\ \\midrule\n')

with open("tables/top_words.tex", "w+") as fp:
    fp.write(latex)
