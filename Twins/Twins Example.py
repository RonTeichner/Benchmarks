#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from causallib.estimation import StratifiedStandardization


# ## Load dataset to a dataframe

# In[2]:


dataset_dict = pd.read_csv('Final_data_twins.csv', index_col=[0])
T = dataset_dict['T'].values
Y = dataset_dict['yf'].values
YCF = dataset_dict['y_cf'].values
all_covariates = pd.DataFrame(dataset_dict.drop(columns=['T', 'y0', 'y1', 'yf', 'y_cf']))
proxies = all_covariates[[col for col in all_covariates if col.startswith('gestat')]]
true_confounders = all_covariates[[col for col in all_covariates if not col.startswith('gestat')]]

y0 = Y * (1 - T) + YCF * T
y1 = Y * T + YCF * (1 - T)
tau = np.mean(y1-y0)
print(f'True Causal Effect: {tau}')

nx = proxies.shape[1]
nu = true_confounders.shape[1]
col_names = ['T', 'Y']
for iR in range(nu):
    col_names += (['U' + str(iR)])
for iX in range(nx):
    col_names += (['X' + str(iX)])

df = pd.DataFrame(np.hstack([T.reshape(-1, 1), Y.reshape(-1, 1), true_confounders, proxies]), columns=col_names)


# ## Estimate causal effect

# In[4]:


std = StratifiedStandardization(LogisticRegression(max_iter=1000))
std.fit(df.copy().drop(columns=['T', 'Y']), df['T'], df['Y'])
pop_outcomes = std.estimate_population_outcome(df.copy().drop(columns=['T', 'Y']), df['T'])
lr = std.estimate_effect(pop_outcomes[1], pop_outcomes[0])
print(f'Estimated Causal Effect: {lr.item()}')

