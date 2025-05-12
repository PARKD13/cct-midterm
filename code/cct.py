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
    



