import pandas as pd 
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("annotation_id")
parser.add_argument("n_tasks", type=int)
parser.add_argument("--weight", action="store_true", default=False)
args = parser.parse_args()

categories = pd.read_csv("analyses/validation.csv").drop_duplicates(["topic", "pacs"])

topics = pd.read_csv("output/hep-ct-75-0.1-0.001-130000-20/descriptions.csv")["description"].tolist()
n_topics = len(topics)
n_top = 10

if args.weight:
    probs = pd.read_parquet("output/hep-ct-75-0.1-0.001-130000-20/topics_0.parquet")[["probs"]]
    probs = np.stack(probs.probs.values)
    probs = np.mean(probs,axis=0)
    weighted = "weighted"
else:
    probs = np.zeros(n_topics)+1
    weighted = "unweighted"

tasks = []

for i in range(args.n_tasks):
    n1 = random.choices(np.arange(n_topics), probs, k=1)[0]
    t1 = topics[n1]
    t2 = ""
    u = categories[categories["topic"] == t1].head(n_top)

    mixture = random.choice([True, False])

    if mixture:
        n2 = random.choices(np.delete(np.arange(n_topics),n1), np.delete(probs,n1), k=1)[0]
        t2 = topics[n2]
        u1 = u.sample(int(n_top/2))
        u2 = categories[(categories["topic"] == t2)].head(n_top)
        u2 = u2[~u2["description"].isin(u1["description"])].sample(int(n_top/2))
        u1 = u1["description"].tolist()
        u2 = u2["description"].tolist()
    else:
        u = u.sample(frac=1)
        u1 = u["description"][:int(n_top/2)].tolist()
        u2 = u["description"][int(n_top/2):int(n_top)].tolist()

    tasks.append({
        'question': i,
        'topic1': t1,
        'topic2': t2,
        'categories1': u1,
        "categories2": u2
    })

tasks = pd.DataFrame(tasks)
tasks.to_csv(f"analyses/truth_{args.annotation_id}_{weighted}.csv")

questions = tasks.copy().set_index("question")[["categories1", "categories2"]]
questions["categories1"] = questions["categories1"].map(lambda l: "\n".join(l))
questions["categories2"] = questions["categories2"].map(lambda l: "\n".join(l))
questions["1 topic or 2 topics ?"] = ""
questions.to_excel(f"analyses/questions_{args.annotation_id}_{weighted}.xlsx", merge_cells=True)
