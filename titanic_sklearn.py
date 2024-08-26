import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from preprocess import load_titanic_data, preprocess_feats

titanic_train, titanic_test = load_titanic_data()

titanic_train.columns = titanic_train.columns.str.lower()
titanic_test.columns = titanic_test.columns.str.lower()

raw_train_feats = titanic_train.drop(columns=["survived", "name"])
raw_test_feats = titanic_test.drop(columns="name")

train_label = titanic_train.survived

train_feats = preprocess_feats(raw_train_feats)
test_feats = preprocess_feats(raw_test_feats)

# SGD
sgd_clf = SGDClassifier(random_state=27, n_jobs=8)

sgd_param_grid = [
    {
        "alpha": np.arange(1e-5, 1e-3, 1e-4),
        "loss": ["log_loss", "hinge", "modified_huber", "squared_hinge"],
        "penalty": ["l2", "l1", "elasticnet"],
    }
]

sgd_cv = GridSearchCV(
    estimator=sgd_clf,
    param_grid=sgd_param_grid,
    scoring="accuracy",
)

sgd_cv.fit(train_feats, train_label)
sgd_cv.best_params_
sgd_cv.best_estimator_

sgd_results = pd.DataFrame(sgd_cv.cv_results_)
sgd_results.sort_values("mean_test_score", ascending=False)  # best_score = 0.836118

# KNN
knn_clf = KNeighborsClassifier()

knn_param_grid = [
    {
        "n_neighbors": range(1, 100, 5),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
]

knn_cv = GridSearchCV(
    estimator=knn_clf,
    param_grid=knn_param_grid,
    scoring="accuracy",
)

knn_cv.fit(train_feats, train_label)
knn_cv.best_params_
knn_cv.best_estimator_

knn_results = pd.DataFrame(knn_cv.cv_results_)
knn_results.sort_values("mean_test_score", ascending=False)  # best_score = 0.823803

# Random Forest
rf_clf = RandomForestClassifier(random_state=42, n_jobs=8)

rf_param_grid = [
    {
        "n_estimators": range(0, 1000, 100),
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": np.arange(0, 0.5, 0.1),
        "oob_score": [True, False],
    }
]

rf_cv = GridSearchCV(
    estimator=rf_clf, param_grid=rf_param_grid, scoring="accuracy", verbose=10
)

rf_cv.fit(train_feats, train_label)
rf_cv.best_params_
rf_cv.best_estimator_

rf_results = pd.DataFrame(rf_cv.cv_results_)
rf_results.sort_values("mean_test_score", ascending=False)  # best_score = 0.820419
