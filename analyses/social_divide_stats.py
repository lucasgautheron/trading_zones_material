import pandas as pd 
import numpy as np
from scipy.stats import beta

q = 0.2

fit = pd.read_parquet("output/social_divide_samples.parquet")

results = []
for i, cat in enumerate(['th','exp','ph']):
    alphas = fit[f'alphas.{i+1}'].values
    betas = fit[f'betas.{i+1}'].values
    lows = np.zeros(len(fit))
    highs = np.zeros(len(fit))
    
    for j in range(len(fit)):
        lows[j] = beta.cdf(q, alphas[j], betas[j])
        highs[j] = 1-beta.cdf(1-q, alphas[j], betas[j])

    print(lows)
    print(highs)

    results.append({
        'cat': cat,
        'low': np.mean(lows),
        'high': np.mean(highs),
        'alpha': np.mean(alphas),
        'beta': np.mean(betas)
    })

results = pd.DataFrame(results)
results['total'] = results['low']+results['high']

results.to_csv("models/social_divide/entrenchment.csv")
print(results)
