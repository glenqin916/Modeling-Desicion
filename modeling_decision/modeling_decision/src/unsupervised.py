"""R4: k-means on the comorbidity block, fit on train only.

Two places it can change the decision:
  1. Do the model's misses concentrate in one cluster? (reported in slices, exploratory)
  2. Does giving the simple model the cluster id raise precision at 40 in grouped CV on train?
     The choice is made on CV, never on the test set.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config
from .data import COMORB


def choose_k(train: pd.DataFrame, sample=5000):
    X = train[COMORB].to_numpy(float)
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    rows = []
    for k in config.KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[idx], km.labels_[idx])})
    table = pd.DataFrame(rows)
    return int(table.loc[table["silhouette"].idxmax(), "k"]), table


def fit_clusters(train: pd.DataFrame, k: int):
    return KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(train[COMORB].to_numpy(float))


def assign(km, df: pd.DataFrame) -> pd.Series:
    return pd.Series(km.predict(df[COMORB].to_numpy(float)), index=df.index).astype(str)


def profile(train: pd.DataFrame) -> pd.DataFrame:
    """Cluster size, comorbidity prevalence and train readmission rate, for naming clusters."""
    g = train.groupby("comorbidity_cluster")
    prof = g[COMORB].mean().round(2)
    prof.insert(0, "n_train", g.size())
    prof.insert(1, "readmit_rate_train", g["y"].mean().round(3))
    return prof
