"""Model definitions and Bayesian search spaces (owner: Person 2).

Task framing: binary classification of (GPS observation, candidate arc)
pairs - "is this candidate the reference arc of this observation?".  The
predicted probability is used (a) for observation-level Top-1 selection and
(b) as the HMM emission score, so probability quality matters, not only the
0/1 decision.

All estimators that are sensitive to feature scale are wrapped in a
``Pipeline(StandardScaler, model)`` so the scaler is fitted on training folds
only (no leakage into validation folds or student-test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[[dict, int, int], Any]       # (params, seed, n_jobs) -> estimator
    default_params: dict
    search_space: Callable[[Any], dict] | None = None   # optuna trial -> params
    slow: bool = False                            # fit on a capped subsample
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _scaled(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


# ------------------------------------------------------------------ builders
def _logreg(p, seed, n_jobs):
    return _scaled(LogisticRegression(C=p["C"], class_weight=p.get("class_weight"), max_iter=2000,
                                      random_state=seed))


def _gnb(p, seed, n_jobs):
    return _scaled(GaussianNB(var_smoothing=p["var_smoothing"]))


def _knn(p, seed, n_jobs):
    return _scaled(KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=n_jobs))


def _rf(p, seed, n_jobs):
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                  min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                  n_jobs=n_jobs, random_state=seed)


def _et(p, seed, n_jobs):
    return ExtraTreesClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                n_jobs=n_jobs, random_state=seed)


def _hgb(p, seed, n_jobs):
    return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_iter=p["max_iter"],
                                          max_leaf_nodes=p["max_leaf_nodes"],
                                          min_samples_leaf=p["min_samples_leaf"],
                                          l2_regularization=p["l2_regularization"],
                                          early_stopping=False, random_state=seed)


def _lgbm(p, seed, n_jobs):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                          num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                          subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                          reg_lambda=p["reg_lambda"], random_state=seed, n_jobs=n_jobs, verbose=-1)


def _mlp(p, seed, n_jobs):
    return _scaled(MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), alpha=p["alpha"],
                                 learning_rate_init=p["learning_rate_init"], batch_size=512,
                                 max_iter=p["max_iter"], early_stopping=True, validation_fraction=0.1,
                                 n_iter_no_change=8, random_state=seed))


# ------------------------------------------------------------- search spaces
def _ss_logreg(t):
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True),
            "class_weight": t.suggest_categorical("class_weight", [None, "balanced"])}


def _ss_gnb(t):
    return {"var_smoothing": t.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)}


def _ss_knn(t):
    return {"n_neighbors": t.suggest_int("n_neighbors", 5, 101, step=2),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"])}


def _ss_trees(t):
    # Bounded on purpose: unlimited-depth forests with hundreds of trees were
    # too slow to train inside CV and to serve from the API (inference cost).
    return {"n_estimators": t.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": t.suggest_categorical("max_depth", [8, 12, 16, 20, 24]),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 50, log=True),
            "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}


def _ss_hgb(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 300, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True)}


def _ss_lgbm(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}


def _ss_mlp(t):
    arch = t.suggest_categorical("arch", ["32", "64", "64-32", "128-64", "128-64-32"])
    return {"hidden_layer_sizes": [int(v) for v in arch.split("-")],
            "alpha": t.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 200}


MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", "Linear / probabilistic", _logreg,
                        {"C": 1.0, "class_weight": None}, _ss_logreg,
                        notes="Linear decision boundary on standardised features"),
    "gnb": ModelSpec("gnb", "Probabilistic (generative)", _gnb, {"var_smoothing": 1e-9}, _ss_gnb, slow=True,
                     notes="Assumes conditionally independent Gaussian features"),
    "knn": ModelSpec("knn", "Instance-based", _knn, {"n_neighbors": 25, "weights": "distance"}, _ss_knn,
                     slow=True, notes="Distance-weighted KNN on standardised features; fit on capped subsample"),
    "random_forest": ModelSpec("random_forest", "Ensemble (bagging)", _rf,
                               {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                                "max_features": "sqrt"}, _ss_trees),
    "extra_trees": ModelSpec("extra_trees", "Ensemble (randomised trees)", _et,
                             {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                              "max_features": "sqrt"}, _ss_trees),
    "hist_gb": ModelSpec("hist_gb", "Boosting", _hgb,
                         {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31,
                          "min_samples_leaf": 20, "l2_regularization": 0.0}, _ss_hgb),
    "lightgbm": ModelSpec("lightgbm", "Boosting", _lgbm,
                          {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                           "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                           "reg_lambda": 0.0}, _ss_lgbm),
    "mlp": ModelSpec("mlp", "Neural network", _mlp,
                     {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "learning_rate_init": 1e-3,
                      "max_iter": 200}, _ss_mlp, slow=True,
                     notes="Shallow MLP with early stopping; produces epoch-wise loss curve"),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[name]


def build_model(name: str, params: dict | None = None, seed: int = 42, n_jobs: int = -1):
    spec = get_spec(name)
    p = dict(spec.default_params)
    p.update(params or {})
    return spec.build(p, seed, n_jobs), p
