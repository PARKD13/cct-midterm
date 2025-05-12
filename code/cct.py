#Cultural Concensus Theory

import pandas as pd
import numpy as np
import pymc as pm


def load_data(filepath):
    #function to load data from csv file
    df = pd.read_csv(filepath)
    data = df.drop(columns=["Informant"], errors='ignore').values
    return data

X = load_data("/host/data/plant_knowledge.csv")  # shape: (N, M)
N, M = X.shape
#Implementation in PyMC
with pm.Model() as cct_model:
    # Priors
    D = pm.Uniform("D", lower=0.5, upper=1, shape=N)
    Z = pm.Bernoulli("Z", p=0.5, shape=M)
    
    # Reshape for broadcasting
    D_reshaped = D[:, None]  # (N, 1)
    Z_reshaped = Z[None, :]  # (1, M)

    # Compute response probabilities
    p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)
    # Likelihood
    X_obs = pm.Bernoulli("X_obs", p=p, observed=X)


