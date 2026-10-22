"""Modèles et validation croisée groupée.

Trois modèles, chacun dans un Pipeline scikit-learn (toute la préparation est ajustée sur le train du fold uniquement) :
  - DummyRegressor : prédit toujours la moyenne du train -> référence plancher ;
  - Ridge          : régression linéaire régularisée (imputation médiane + standardisation) ;
  - XGBRegressor   : arbres de décision boostés (gère nativement les valeurs manquantes).
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config as C

N_SPLITS = 5


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# MAE = métrique principale ; RMSE et R² = garde-fous
SCORING = {"MAE": make_scorer(mean_absolute_error, greater_is_better=False),
           "RMSE": make_scorer(rmse, greater_is_better=False),
           "R2": "r2"}


def cv_splitter():
    """Validation croisée groupée : tous les séismes d'une même heure de bruit restent dans le même fold."""
    return GroupKFold(n_splits=N_SPLITS)


def make_dummy():
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def make_ridge(alpha=1.0):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", Ridge(alpha=alpha))])


def make_xgb(**params):
    base = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, reg_lambda=1.0, tree_method="hist", n_jobs=-1,
                random_state=C.RANDOM_STATE)
    base.update(params)
    return Pipeline([("model", XGBRegressor(**base))])


def cv_report(model, X, y, groups, name):
    """Validation croisée groupée : moyenne ± écart-type, en entraînement ET en validation."""
    res = cross_validate(model, X, y, groups=groups, cv=cv_splitter(), scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    row = {"modèle": name}
    for m in SCORING:
        sign = -1 if m != "R2" else 1
        tr, va = sign * res[f"train_{m}"], sign * res[f"test_{m}"]
        row[f"{m} train"] = tr.mean()
        row[f"{m} val (moy)"] = va.mean()
        row[f"{m} val (é.-t.)"] = va.std()
    row["temps ajust. (s)"] = res["fit_time"].mean()
    return row


def scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}
