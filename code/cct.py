#Cultural Concensus Theory

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

#read and load data
df = pd.read_csv("/host/data/plant_knowledge.csv".strip())
data = df.drop(columns=["Informant"]).to_numpy(dtype=int)
N, M = data.shape

#implementation in PyMC
with pm.Model() as cct_model:
    # Priors
    D = pm.Uniform("D", lower=0.5, upper=1, shape=N)
    Z = pm.Bernoulli("Z", p=0.5, shape=M)
    
    #reshape for broadcasting
    D_reshaped = D[:, None]  # (N, 1)
    
    #compute pij
    p = Z * D_reshaped + (1-Z) * (1-D_reshaped)

    #likelihood
    X_obs = pm.Bernoulli("X_obs", p=p, observed=data)

    #inference
    trace = pm.sample(draws=2000,
                        tune=1000,
                        chains=4,
                        target_accept=0.9,
                        return_inferencedata=True)
    
    #1. summarize samples
    summary = az.summary(trace, var_names=['D', 'Z'])
    print(summary)

    #2. estimate informant competence
    competence_means = trace.posterior['D'].mean(dim=['chain', 'draw']).values
    #plot competence_means
    az.plot_posterior(trace, var_names=['D'])

    most_competent = np.argmax(competence_means)
    least_competent = np.argmin(competence_means)

    print(f'\nMost competent Informant: {most_competent+1}')
    print(f'Least competent Informant: {least_competent+1}\n')

    #3. estimate concensus answers
    Z_means = trace.posterior['Z'].mean(dim=['chain', 'draw']).values
    Z_mode = (Z_means > 0.5).astype(int)

    print(f'Posterior mean probabilty for Zj: {Z_means}')
    print(f'Most likely concensus answer key(Mode): {Z_mode}\n')

    az.plot_posterior(trace, var_names=['Z'])

    #4. Compare with Naive Aggregation
    majority_vote = (np.mean(data, axis=0) > 0.5).astype(int)

    print(f'Naive majority vote answer key: {majority_vote}')
    print(f'CCT concensus answer key: {Z_mode}')
    



