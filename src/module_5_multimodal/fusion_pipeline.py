# =============================================================================
# MODULE 5 — PIPELINE MULTIMODAL (VERSION CORRIGÉE + MLflow)
# Projet : Eco-Smart Classifier
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import joblib
import warnings
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, mean_squared_error, r2_score,
                              mean_absolute_error)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier, XGBRegressor

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.module_4_NLP.text_classifier import preprocess_text

warnings.filterwarnings("ignore")

# Chemins
CLEAN_DATA_PATH = "data/processed/clean_data.csv"
PROCESSED_DIR   = "data/processed"
MODELS_DIR      = "models"
FIGURES_DIR     = "reports/figures"
NLP_MODELS_DIR  = os.path.join(MODELS_DIR, "nlp")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(NLP_MODELS_DIR, exist_ok=True)

ORIG_NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]


def build_multimodal_features(df: pd.DataFrame, scaler, ohe, tfidf_vec) -> np.ndarray:
    """Construit les features multimodales."""
    df = df.copy()
    df["Source"] = df["Source"].fillna("Inconnu")
    num_scaled = scaler.transform(df[ORIG_NUM_COLS])
    num_no_price = num_scaled[:, :-1]
    src_encoded = ohe.transform(df[["Source"]])
    src_array = src_encoded.toarray() if hasattr(src_encoded, "toarray") else src_encoded
    texts_clean = [preprocess_text(str(t)) for t in df["Rapport_Collecte"].values]
    text_vec = tfidf_vec.transform(texts_clean).toarray()
    X_final = np.hstack([num_no_price, src_array, text_vec])
    return X_final


def evaluate_model(model, X_test, y_test, model_name: str, task: str = "classification"):
    """Évalue un modèle et log les résultats."""
    if task == "classification":
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"  {model_name:30s} — Acc: {acc:.4f} | F1: {f1:.4f}")
        mlflow.log_metric(f"{model_name}_accuracy", acc)
        mlflow.log_metric(f"{model_name}_f1_weighted", f1)
        return acc, f1, y_pred
    elif task == "regression":
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  {model_name:30s} — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        mlflow.log_metric(f"{model_name}_rmse", rmse)
        mlflow.log_metric(f"{model_name}_mae", mae)
        mlflow.log_metric(f"{model_name}_r2", r2)
        return rmse, mae, r2, y_pred


def run_multimodal_pipeline(experiment_name: str = "Eco-Smart Multimodal Pipeline"):
    """Pipeline multimodal complet."""
    
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name="Multimodal_Full_Pipeline")
    
    print("=" * 60)
    print("MODULE 5 — PIPELINE MULTIMODAL (avec MLflow)")
    print("=" * 60)

    # 1. CHARGEMENT
    print("\n1. CHARGEMENT DES DONNÉES")
    print("-" * 40)
    df = pd.read_csv(CLEAN_DATA_PATH)
    df_labeled = df.dropna(subset=["Categorie_enc"]).copy()
    print(f"Lignes avec label : {df_labeled.shape[0]}")
    print("\nDistribution des catégories:")
    print(df_labeled["Categorie"].value_counts())

    # 2. ENCODEURS
    print("\n2. CHARGEMENT DES ENCODEURS")
    print("-" * 40)
    scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))
    ohe = joblib.load(os.path.join(PROCESSED_DIR, "onehot_encoder.pkl"))
    le = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
   
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        tfidf_vec = joblib.load(tfidf_path)
        print(f"  ✓ TF-IDF vectorizer chargé")
    else:
        raise FileNotFoundError("Vectoriseur TF-IDF non trouvé !")

    # 3. FEATURES
    print("\n3. CONSTRUCTION DES FEATURES")
    print("-" * 40)
    X = build_multimodal_features(df_labeled, scaler, ohe, tfidf_vec)
    y = df_labeled["Categorie_enc"].astype(int).values
    y_reg = df_labeled["Prix_Revente"].values
    print(f"Shape X: {X.shape}")

    # 4. SPLIT
    print("\n4. SPLIT TRAIN / VAL / TEST")
    print("-" * 40)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(sss1.split(X, y))
    X_tv, X_te = X[tv_idx], X[te_idx]
    y_tv, y_te = y[tv_idx], y[te_idx]
    y_reg_tv, y_reg_te = y_reg[tv_idx], y_reg[te_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    tr_idx, va_idx = next(sss2.split(X_tv, y_tv))
    X_train, X_val = X_tv[tr_idx], X_tv[va_idx]
    y_train, y_val = y_tv[tr_idx], y_tv[va_idx]
    y_reg_train, y_reg_val = y_reg_tv[tr_idx], y_reg_tv[va_idx]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

    # 5. CLASSIFICATION
    print("\n5. CLASSIFICATION MULTIMODALE")
    print("=" * 60)

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=15, 
            min_samples_split=5, min_samples_leaf=2, class_weight="balanced", 
            random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    }

    best_clf = None
    best_f1 = 0.0
    best_name = ""

    print("\nValidation :")
    for name, clf in classifiers.items():
        print(f"\n  --- {name} ---")
        clf.fit(X_train, y_train)
        y_pred_val = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, average="weighted")
        
        print(f"    Validation — Accuracy: {acc:.4f} | F1: {f1:.4f}")
        mlflow.log_metric(f"{name}_val_accuracy", acc)
        mlflow.log_metric(f"{name}_val_f1", f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_clf = clf
            best_name = name

    print(f"\n  ✓ Meilleur classifieur : {best_name} (F1: {best_f1:.4f})")
    mlflow.log_param("best_classifier", best_name)

    # Test classification
    print("\nTest classification :")
    best_y_pred = best_clf.predict(X_te)
    test_acc = accuracy_score(y_te, best_y_pred)
    test_f1 = f1_score(y_te, best_y_pred, average="weighted")
    print(f"  ✓ Test — Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    mlflow.log_metric("best_classifier_test_accuracy", test_acc)
    mlflow.log_metric("best_classifier_test_f1", test_f1)

    print("\nRapport de classification détaillé:")
    print(classification_report(y_te, best_y_pred, target_names=le.classes_))
    
    # Matrice de confusion
    cm = confusion_matrix(y_te, best_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Matrice de Confusion - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig13_multimodal_confusion_matrix.png"), dpi=120)
    plt.close()
    print("  ✓ fig13_multimodal_confusion_matrix.png sauvegardé")

    # 6. RÉGRESSION
    print("\n6. RÉGRESSION MULTIMODALE")
    print("=" * 60)

    pos_mask_train = y_reg_train >= 0
    pos_mask_val = y_reg_val >= 0
    pos_mask_test = y_reg_te >= 0
   
    X_reg_train, y_reg_train_clean = X_train[pos_mask_train], y_reg_train[pos_mask_train]
    X_reg_val, y_reg_val_clean = X_val[pos_mask_val], y_reg_val[pos_mask_val]
    X_reg_test, y_reg_test_clean = X_te[pos_mask_test], y_reg_te[pos_mask_test]
   
    print(f"  Prix négatifs filtrés: Train={np.sum(~pos_mask_train)}, Test={np.sum(~pos_mask_test)}")

    reg = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                random_state=42, n_jobs=-1)
    reg.fit(X_reg_train, y_reg_train_clean)
   
    y_pred_reg = reg.predict(X_reg_test)
    y_pred_reg = np.maximum(y_pred_reg, 0)
   
    rmse = np.sqrt(mean_squared_error(y_reg_test_clean, y_pred_reg))
    r2 = r2_score(y_reg_test_clean, y_pred_reg)
    print(f"Test — RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # 7. SAUVEGARDE
    print("\n7. SAUVEGARDE")
    print("-" * 40)
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "multimodal_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "multimodal_regressor.pkl"))
    joblib.dump(tfidf_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("  ✓ Modèles sauvegardés")

    mlflow.end_run()
    
    print("\n" + "=" * 60)
    print("MODULE 5 TERMINÉ ✓")
    print("=" * 60)

    return best_clf, reg


if __name__ == "__main__":
    run_multimodal_pipeline()