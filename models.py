#import torch
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import torch
import math
import numpy as np
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

def get_model(name):
    if name == "linear":
        return Linear()
    elif name == "mean":
        return MeanModel()
    elif name == "rf":
        return RandomForest()
    elif name == "mlp":
        return MLP()
    else:
        raise ValueError(f"Model kind {name} not implemented")
    

class MeanModel:
    def fit(self, X_train, y_train):
        y_train = y_train.astype(int)
        labels = set(y_train)
        means = []
        for label in labels:
            means.append(X_train[y_train == label].mean(axis=0))
        self.means = means

    def predict_proba(self, X):
        mean_diffs = []
        for mean in self.means:
            mean_diffs.append(np.linalg.norm(X - mean, axis=1))
        mean_diffs = np.array(mean_diffs).T # has shape (n_samples, n_labels)
        p = (mean_diffs.T / mean_diffs.sum(axis=1)).T
        return p
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def save(self, path, name="model"):
        for i, mean in enumerate(self.means):
            np.save(path+f"/{name}_{i}.npy", mean)

    def load(self, path, name="model"):
        self.means = []
        i = 0
        while True:
            try:
                self.means.append(np.load(path+f"/{name}_{i}.npy"))
                i += 1
            except FileNotFoundError:
                break
        if len(self.means) == 0:
            raise ValueError("No means found in path")


class SKLearnModel:
    def __init__(self, scale=True):
        self.scale = scale
        if scale:
            self.scaler = StandardScaler()


    def fit(self, X_train, y_train):
        if self.scale:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def score(self, X_train, y_train):
        if self.scale:
            X_train = self.scaler.transform(X_train)
        return self.model.score(X_train, y_train)
    
    def predict_proba(self, X):
        if self.scale:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)
        return proba
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    def save(self, path, name="model"):
        with open(path+f"/{name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path, name="model"):
        with open(path+f"/{name}.pkl", "rb") as f:
            self.model = pickle.load(f) 

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name


class Linear(SKLearnModel):
    def __init__(self, penalty='l2', C=1.0):
        super().__init__()
        self.name = f"linear-{penalty}-C-{C}"
        self.model = LogisticRegression(random_state=0, penalty=penalty, C=C, class_weight="balanced")

    def save(self, path, name="model"):
        with open(path+f"/{name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.save_weight(path, name=name)

    def save_weight(self, path, name="model"):
        np.save(path+f"/{name}.npy", self.model.coef_)



class RandomForest(SKLearnModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.name = f"rf-n-{n_estimators}-d-{max_depth}"
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


class MLP(SKLearnModel):
    def __init__(self, hidden_layer_sizes=(1000, 500), max_iter=200):
        super().__init__()
        self.name = f"mlp-hl-{hidden_layer_sizes}-mi-{max_iter}"
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)