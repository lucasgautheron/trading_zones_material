import pandas as pd 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("output/hep-ct-validation/lda_tuning_results.csv")
df['overfitting'] = (df['perplexity']/df['train_perplexity']).apply(np.log)

df = df[df['topics'] > 25]
df = df[df['overfitting'] < 1]

df.sort_values('c_npmi', ascending=False, inplace=True)
print(df)
print(len(df))
