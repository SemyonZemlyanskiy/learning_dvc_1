import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

df = pd.read_csv('data/diabet.csv')
X = df.drop('target', axis = 1)
y = df['target']

iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(X)
mask = outliers != -1  
X, y = X[mask], y[mask] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_prepared = pd.DataFrame(data=X_scaled)
df_prepared['target'] = y
df_prepared = df_prepared.dropna(axis=0)

df_prepared.to_csv('data/prepared.csv')