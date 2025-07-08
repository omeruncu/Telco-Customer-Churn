import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import os


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, save_model=False, model_path=None):
    """
    Verilen modeli eğitir, test seti üzerinde değerlendirir ve opsiyonel olarak kaydeder.

    Parameters:
        model: sklearn model objesi (tuned veya default)
        model_name: string, modelin adı (örneğin "Logistic Regression (Tuned)")
        save_model: bool, True ise modeli .pkl olarak kaydeder
        model_path: string, modelin kaydedileceği yol

    Returns:
        model: Eğitilmiş model
    """
    print(f"\n🔍 Model: {model_name}")
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Classification Report
    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", color="darkorange")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    # Modeli kaydet
    if save_model:
        path = model_path or f"models/{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 🔹 Klasörü oluştur
        joblib.dump(model, path)
        print(f"💾 Model kaydedildi: {path}")

    return model

from sklearn.ensemble import VotingClassifier

def build_soft_voting_classifier(best_models):
    """
    En iyi 3 modeli soft voting ile birleştirir.
    """
    voting_clf = VotingClassifier(
        estimators=[
            ("catboost", best_models["CatBoost (Tuned)"]),
            ("logreg", best_models["Logistic Regression (Tuned)"]),
            ("lightgbm", best_models["LightGBM (Tuned)"])
        ],
        voting="soft"
    )
    return voting_clf

