from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

def evaluate_models_with_optimization(X, y, cv=5, random_state=42):
    """
    Varsayılan ve optimize edilmiş modelleri çoklu metriklerle değerlendirir.
    Returns:
        pd.DataFrame: Model adı ve metrik skorlarını içeren tablo
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state)
    }

    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5]},
        "CatBoost": {"depth": [4, 6], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
    }

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }

    rows = []
    best_models = {}

    for name, model in models.items():
        print(f"\n🔹 {name} (Default)")
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "Model": f"{name} (Default)",
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC AUC": scores["test_roc_auc"].mean()
        }
        rows.append(row)

        print(f"🔧 {name} (Tuned)")
        grid = GridSearchCV(model, param_grids[name], cv=cv, scoring="roc_auc", n_jobs=1)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        best_models[f"{name} (Tuned)"] = best_model
        tuned_scores = cross_validate(best_model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "Model": f"{name} (Tuned)",
            "Accuracy": tuned_scores["test_accuracy"].mean(),
            "Precision": tuned_scores["test_precision"].mean(),
            "Recall": tuned_scores["test_recall"].mean(),
            "F1": tuned_scores["test_f1"].mean(),
            "ROC AUC": tuned_scores["test_roc_auc"].mean()
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df.sort_values(by="ROC AUC", ascending=False).reset_index(drop=True), best_models
