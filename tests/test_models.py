# =============================================================================
# tests/test_model.py
# Tests du Module 2 : classificateur et régresseur
# =============================================================================

from matplotlib.pyplot import clf
import pytest
import numpy as np
import os
import scipy.sparse as sp

from sklearn.metrics import f1_score, accuracy_score

import joblib, pandas as pd

# =============================================================================
# 1. TESTS SUR LE MODÈLE MULTIMODAL (artefacts fitté via conftest)
# =============================================================================

class TestClassifierWithSyntheticData:
    """
    Tests utilisant le classifieur léger entraîné sur données synthétiques
    (via la fixture fitted_classifier de conftest.py).
    Ces tests sont TOUJOURS stables, même avant l'entraînement réel.
    """

    def test_classifier_predict_output_shape(
        self, fitted_classifier, fitted_scaler, fitted_ohe, fitted_tfidf, labeled_data
    ):
        """La prédiction doit retourner autant de labels que d'entrées."""
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        df = labeled_data.fillna(labeled_data[num_cols].median()).head(20)

        num_scaled = fitted_scaler.transform(df[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(df[["Source"]])
        texts = [preprocess_text(t) for t in df["Rapport_Collecte"].astype(str)]
        text_vec = fitted_tfidf.transform(texts)

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])
        preds = fitted_classifier.predict(X)

        assert preds.shape[0] == 20

    def test_classifier_predictions_are_valid_classes(
        self, fitted_classifier, fitted_scaler, fitted_ohe, fitted_tfidf,
        fitted_le, labeled_data
    ):
        """Les prédictions doivent appartenir aux classes connues."""
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        df = labeled_data.fillna(labeled_data[num_cols].median()).head(10)

        num_scaled = fitted_scaler.transform(df[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(df[["Source"]])
        texts = [preprocess_text(t) for t in df["Rapport_Collecte"].astype(str)]
        text_vec = fitted_tfidf.transform(texts)

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])
        preds = fitted_classifier.predict(X)

        valid_classes = set(range(len(fitted_le.classes_)))
        for pred in preds:
            assert int(pred) in valid_classes, f"Classe inconnue prédite : {pred}"

    def test_classifier_f1_above_chance(
        self, fitted_classifier, fitted_scaler, fitted_ohe, fitted_tfidf, labeled_data
    ):
        """
        Le classifieur synthétique doit faire mieux que le hasard total.
        Seuil volontairement bas (données aléatoires, peu d'exemples).
        """
        from sklearn.model_selection import train_test_split
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        df = labeled_data.fillna(labeled_data[num_cols].median())

        num_scaled = fitted_scaler.transform(df[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(df[["Source"]])
        texts = [preprocess_text(t) for t in df["Rapport_Collecte"].astype(str)]
        text_vec = fitted_tfidf.transform(texts)

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])
        y = df["Categorie_enc"].astype(int).values

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = fitted_classifier.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        # Seuil minimal : 10% (données synthétiques aléatoires)
        assert f1 >= 0.10, f"F1 trop faible : {f1:.4f}"

    def test_regressor_predict_output(
        self, fitted_regressor, fitted_scaler, fitted_ohe, fitted_tfidf, labeled_data
    ):
        """Le régresseur doit retourner des valeurs numériques."""
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        df = labeled_data.fillna(labeled_data[num_cols].median()).head(10)

        num_scaled = fitted_scaler.transform(df[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(df[["Source"]])
        texts = [preprocess_text(t) for t in df["Rapport_Collecte"].astype(str)]
        text_vec = fitted_tfidf.transform(texts)

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])
        preds = fitted_regressor.predict(X)

        assert preds.shape[0] == 10
        assert np.all(np.isfinite(preds)), "Prédictions non finies"


# =============================================================================
# 2. TESTS SUR LES MODÈLES RÉELS ENTRAÎNÉS (si disponibles)
# =============================================================================

class TestRealModels:
    """
    Tests sur les vrais modèles sauvegardés après entraînement.
    Ces tests sont skippés si les fichiers ne sont pas présents.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_models(self):
        if not os.path.exists("models/best_classifier.pkl"):
            pytest.skip("Modèle réel non disponible — lancez run_modeling() d'abord")

    def test_best_classifier_exists(self):
        """best_classifier.pkl doit exister après entraînement."""
        assert os.path.exists("models/best_classifier.pkl")

    def test_best_regressor_exists(self):
        """best_regressor.pkl doit exister après entraînement."""
        assert os.path.exists("models/best_regressor.pkl")

    def test_multimodal_classifier_exists(self):
        """multimodal_classifier.pkl doit exister après entraînement."""
        assert os.path.exists("models/multimodal_classifier.pkl")

    def test_model_performance_on_real_test_set(self):

        clf = joblib.load("models/multimodal_classifier.pkl")
        vec = joblib.load("models/tfidf_vectorizer.pkl")
        scaler = joblib.load("data/processed/scaler.pkl")
        ohe = joblib.load("data/processed/onehot_encoder.pkl")

        assert os.path.exists("data/processed/X_test.csv")
        assert os.path.exists("data/processed/y_test.npy")

        X_test_raw = pd.read_csv("data/processed/X_test.csv")
        y_test = np.load("data/processed/y_test.npy")

        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]

        if all(c in X_test_raw.columns for c in num_cols):
            num_scaled = scaler.transform(X_test_raw[num_cols])[:, :-1]
        else:
            num_scaled = X_test_raw.values[:, :5]

        if "Source" in X_test_raw.columns:
            src_enc = ohe.transform(X_test_raw[["Source"]])
            src_sparse = sp.csr_matrix(
            src_enc.toarray() if hasattr(src_enc, "toarray") else src_enc
        )
        else:
            src_sparse = sp.csr_matrix(
            X_test_raw[[c for c in X_test_raw.columns if "Source_" in c]].values
        )

        if "Rapport_Collecte" in X_test_raw.columns:
            texts = [preprocess_text(t) for t in X_test_raw["Rapport_Collecte"].astype(str)]
            text_vec = vec.transform(texts)  # ← utilise le vrai vectorizer sauvegardé
        else:
        # Reconstruire une matrice vide avec le bon nombre de features
          n_text_features = clf.n_features_in_ - num_scaled.shape[1] - src_sparse.shape[1]
        text_vec = sp.csr_matrix((len(X_test_raw), n_text_features))

        X_final = sp.hstack([sp.csr_matrix(num_scaled), src_sparse, text_vec], format="csr")

         # Vérification de cohérence avant prédiction
        assert X_final.shape[1] == clf.n_features_in_, (
             f"Mismatch features : X={X_final.shape[1]}, modèle={clf.n_features_in_}. "
            "Relancez run_modeling() pour régénérer X_test.csv avec le bon pipeline."
        )

        preds = clf.predict(X_final)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        assert f1 >= 0.70, f"F1={f1:.4f} < 0.70"

# =============================================================================
# 3. TESTS DE ROBUSTESSE DES PRÉDICTIONS
# =============================================================================

class TestPredictionRobustness:

    def test_single_sample_prediction(
        self, fitted_classifier, fitted_scaler, fitted_ohe, fitted_tfidf, labeled_data
    ):
        """La prédiction sur un seul exemple doit fonctionner."""
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        row = labeled_data.fillna(labeled_data[num_cols].median()).iloc[[0]]

        num_scaled = fitted_scaler.transform(row[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(row[["Source"]])
        text = preprocess_text(row["Rapport_Collecte"].iloc[0])
        text_vec = fitted_tfidf.transform([text])

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])
        pred = fitted_classifier.predict(X)

        assert pred.shape == (1,)
        assert isinstance(int(pred[0]), int)

    def test_predictions_are_deterministic(
        self, fitted_classifier, fitted_scaler, fitted_ohe, fitted_tfidf, labeled_data
    ):
        """Les prédictions doivent être reproductibles (pas de randomness)."""
        from src.module_4_NLP.text_classifier import preprocess_text

        num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
        df = labeled_data.fillna(labeled_data[num_cols].median()).head(5)

        num_scaled = fitted_scaler.transform(df[num_cols])[:, :-1]
        src_enc = fitted_ohe.transform(df[["Source"]])
        texts = [preprocess_text(t) for t in df["Rapport_Collecte"].astype(str)]
        text_vec = fitted_tfidf.transform(texts)

        X = sp.hstack([sp.csr_matrix(num_scaled), sp.csr_matrix(src_enc), text_vec])

        preds1 = fitted_classifier.predict(X)
        preds2 = fitted_classifier.predict(X)
        np.testing.assert_array_equal(preds1, preds2)