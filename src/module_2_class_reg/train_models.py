# =============================================================================
# MODULE 2 — MODÉLISATION SUPERVISÉE (Classification & Régression)
# Projet : Eco-Smart Classifier
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import shap

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
CLEAN_DATA_PATH = "data/processed/clean_data.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colonnes numériques originales utilisées lors du fit du scaler
ORIG_NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]


# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES FEATURES
# =============================================================================

def load_features(clean_path: str = CLEAN_DATA_PATH):
    """
    Charge le dataset nettoyé, reconstruit les features (num scalées + OHE),
    et retourne X, y_class, y_reg ainsi que les noms de features.
    """
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES ET RECONSTRUCTION DES FEATURES")
    print("=" * 60)

    df = pd.read_csv(clean_path)
    print(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Recharger les encodeurs
    scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))
    ohe = joblib.load(os.path.join(PROCESSED_DIR, "onehot_encoder.pkl"))
    le = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder.pkl"))

    # Features numériques scalées (on enlève Prix_Revente pour éviter la fuite)
    df_num_all = pd.DataFrame(
        scaler.transform(df[ORIG_NUM_COLS]),
        columns=[c + "_scaled" for c in ORIG_NUM_COLS],
        index=df.index,
    )
    df_num = df_num_all.drop(columns=["Prix_Revente_scaled"])

    # One-hot encoding de Source
    src_encoded = ohe.transform(df[["Source"]])
    src_cols = ohe.get_feature_names_out(["Source"])
    df_src = pd.DataFrame(src_encoded, columns=src_cols, index=df.index)

    X = pd.concat([df_num, df_src], axis=1)
    print(f"Features shape : {X.shape}")

    # Sauvegarder les noms de features pour l'API
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "num_feature_names.pkl"))

    y_class = df["Categorie_enc"].values
    y_reg = df["Prix_Revente"].values

    return X, y_class, y_reg, le


# =============================================================================
# 2. SPLIT STRATIFIÉ 70 / 15 / 15
# =============================================================================

def make_splits(X, y_class, y_reg):
    """
    Effectue un split stratifié 70 / 15 / 15.
    Uniquement sur les lignes avec label (y_class non NaN).
    """
    mask = ~np.isnan(y_class.astype(float))
    X_l, yc_l, yr_l = X[mask], y_class[mask].astype(int), y_reg[mask]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(sss1.split(X_l, yc_l))

    X_tv, X_te = X_l.iloc[tv_idx], X_l.iloc[te_idx]
    yc_tv, yc_te = yc_l[tv_idx], yc_l[te_idx]
    yr_tv, yr_te = yr_l[tv_idx], yr_l[te_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    tr_idx, va_idx = next(sss2.split(X_tv, yc_tv))

    X_train, X_val = X_tv.iloc[tr_idx], X_tv.iloc[va_idx]
    yc_train, yc_val = yc_tv[tr_idx], yc_tv[va_idx]
    yr_train, yr_val = yr_tv[tr_idx], yr_tv[va_idx]

    print(f"\nTrain : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_te.shape[0]}")

    # Sauvegarder les vrais splits pour les tests d'intégration
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
    X_te.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), yc_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), yc_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), yc_te)

    return X_train, X_val, X_te, yc_train, yc_val, yc_te, yr_train, yr_val, yr_te


# =============================================================================
# 3. CLASSIFICATION
# =============================================================================

def run_classification(X_train, X_val, X_test, y_train, y_val, y_test, le):
    """
    Entraîne et compare plusieurs classificateurs.
    Loggue dans MLflow. Retourne le meilleur classifieur.
    """
    print("\n" + "=" * 60)
    print("1. CLASSIFICATION — Prédiction de la catégorie")
    print("=" * 60)

    mlflow.set_experiment("eco_smart_classification")

    models_config = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]},
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        },
    }

    best_clf = None
    best_f1 = 0.0

    for name, cfg in models_config.items():
        print(f"\n--- {name} ---")
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["params"],
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)
            est = gs.best_estimator_

            y_pred_val = est.predict(X_val)
            acc = accuracy_score(y_val, y_pred_val)
            f1 = f1_score(y_val, y_pred_val, average="weighted")

            print(f"  Meilleurs params : {gs.best_params_}")
            print(f"  Val — Accuracy: {acc:.4f} | F1: {f1:.4f}")

            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1_weighted", f1)
            sig = infer_signature(X_train, est.predict(X_train))
            mlflow.sklearn.log_model(
                est,
                artifact_path="model",
                signature=sig,
                registered_model_name=f"Classifier_{name}",
            )

            if f1 > best_f1:
                best_f1 = f1
                best_clf = est

    # Évaluation sur le test
    print("\n" + "=" * 60)
    print("ÉVALUATION DU MEILLEUR CLASSIFIEUR SUR LE TEST")
    print("=" * 60)
    y_pred_test = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")
    print(f"Test — Accuracy : {test_acc:.4f} | F1 : {test_f1:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(le.classes_))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(le.classes_, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Catégorie prédite")
    ax.set_ylabel("Vraie catégorie")
    ax.set_title("Matrice de confusion — meilleur classifieur")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Sauvegarde
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "best_classifier.pkl"))
    print(f"  ✓ best_classifier.pkl sauvegardé (F1 val : {best_f1:.4f})")

    return best_clf


# =============================================================================
# 4. ANALYSE SHAP
# =============================================================================

def run_shap_analysis(clf, X_train, X_test):
    """Génère le SHAP summary plot pour le meilleur classifieur.
    Garantit toujours la création du fichier (image de fallback si SHAP échoue).
    """
    print("\n" + "=" * 60)
    print("2. ANALYSE SHAP (interprétabilité)")
    print("=" * 60)

    shap_path = os.path.join(FIGURES_DIR, "shap_importance.png")

    try:
        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            # Pour RF multi-classe, shap_values est une liste → prendre la moyenne abs
            if isinstance(shap_values, list):
                shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                shap.summary_plot(
                    shap_mean, X_test,
                    feature_names=list(X_test.columns),
                    show=False, plot_type="bar",
                )
            else:
                shap.summary_plot(
                    shap_values, X_test,
                    feature_names=list(X_test.columns),
                    show=False, plot_type="bar",
                )
        else:
            sample_idx = np.random.choice(
                X_train.shape[0], min(100, X_train.shape[0]), replace=False
            )
            explainer = shap.KernelExplainer(
                clf.predict_proba, X_train.iloc[sample_idx]
            )
            shap_values = explainer.shap_values(X_test.iloc[:50])
            shap.summary_plot(
                shap_values, X_test.iloc[:50],
                feature_names=list(X_test.columns),
                show=False, plot_type="bar",
            )

        plt.savefig(shap_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ shap_importance.png sauvegardé")

    except Exception as exc:
        print(f"  ⚠ SHAP a échoué ({exc}). Génération d'un graphe de fallback...")
        plt.close("all")

        # --- Fallback : importances natives du modèle ---
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_).mean(axis=0)
            else:
                importances = np.ones(X_test.shape[1])

            feat_names = list(X_test.columns)
            sorted_idx = np.argsort(importances)[::-1][:20]
            ax.barh(
                [feat_names[i] for i in sorted_idx[::-1]],
                importances[sorted_idx[::-1]],
                color="#4C72B0",
            )
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances (fallback — SHAP indisponible)")
        except Exception as inner:
            ax.text(0.5, 0.5, f"Impossible de calculer les importances\n{inner}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("SHAP indisponible")

        plt.tight_layout()
        plt.savefig(shap_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ shap_importance.png (fallback) sauvegardé dans {shap_path}")

# =============================================================================
# 5. RÉGRESSION
# =============================================================================

def run_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Entraîne et compare plusieurs régresseurs.
    Loggue dans MLflow. Retourne le meilleur régresseur.
    """
    print("\n" + "=" * 60)
    print("3. RÉGRESSION — Prédiction du prix de revente")
    print("=" * 60)

    mlflow.set_experiment("eco_smart_regression")

    models_config = {
        "RandomForestReg": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
        },
        "XGBoostReg": {
            "model": XGBRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
        },
        "LinearRegression": {"model": LinearRegression(), "params": {}},
    }

    best_reg = None
    best_rmse = np.inf

    for name, cfg in models_config.items():
        print(f"\n--- {name} ---")
        with mlflow.start_run(run_name=name):
            if cfg["params"]:
                gs = GridSearchCV(
                    estimator=cfg["model"],
                    param_grid=cfg["params"],
                    cv=3,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    verbose=0,
                )
                gs.fit(X_train, y_train)
                est = gs.best_estimator_
                params = gs.best_params_
            else:
                cfg["model"].fit(X_train, y_train)
                est = cfg["model"]
                params = {}

            y_pred_val = est.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            mae = mean_absolute_error(y_val, y_pred_val)
            r2 = r2_score(y_val, y_pred_val)
            print(f"  Val — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_r2", r2)
            sig = infer_signature(X_train, est.predict(X_train))
            mlflow.sklearn.log_model(
                est,
                artifact_path="model",
                signature=sig,
                registered_model_name=f"Regressor_{name}",
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_reg = est

    y_pred_test = best_reg.predict(X_test)
    rmse_t = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_t = mean_absolute_error(y_test, y_pred_test)
    r2_t = r2_score(y_test, y_pred_test)
    print(f"\nTest — RMSE: {rmse_t:.4f} | MAE: {mae_t:.4f} | R²: {r2_t:.4f}")

    joblib.dump(best_reg, os.path.join(MODELS_DIR, "best_regressor.pkl"))
    print(f"  ✓ best_regressor.pkl sauvegardé (RMSE val : {best_rmse:.4f})")

    return best_reg


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

def run_modeling():
    """Lance le pipeline complet de modélisation supervisée."""
    X, y_class, y_reg, le = load_features()
    (X_train, X_val, X_test,
     yc_train, yc_val, yc_test,
     yr_train, yr_val, yr_test) = make_splits(X, y_class, y_reg)

    best_clf = run_classification(X_train, X_val, X_test,
                                  yc_train, yc_val, yc_test, le)
    run_shap_analysis(best_clf, X_train, X_test)
    run_regression(X_train, X_val, X_test, yr_train, yr_val, yr_test)

    print("\n" + "=" * 60)
    print("MODULE 2 TERMINÉ")
    print("=" * 60)
    # À ajouter à la fin de run_modeling(), avant le print final :
    X_train.to_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), index=False)


if __name__ == "__main__":
    run_modeling()