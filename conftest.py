# =============================================================================
# tests/conftest.py — Fixtures partagées entre tous les tests
# =============================================================================

import pytest
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# =============================================================================
# FIXTURE : DATASET SYNTHÉTIQUE
# =============================================================================

@pytest.fixture(scope="session")
def sample_data():
    np.random.seed(42)
    n = 200

    categories = ["Papier", "Plastique", "Verre", "Métal", "Organique"]
    sources = ["Usine_A", "Usine_B", "Centre_Tri", "Collecte_Citoyenne"]

    cat_labels = np.random.choice(categories, n)
    cat_enc = np.where(
        np.random.rand(n) < 0.15,
        np.nan,
        [categories.index(c) for c in cat_labels]
    )

    data = {
        "Poids": np.where(np.random.rand(n) < 0.10, np.nan,
                          np.random.uniform(5, 200, n)),
        "Volume": np.where(np.random.rand(n) < 0.05,
                           np.random.uniform(1000, 5000, n),
                           np.random.uniform(10, 500, n)),
        "Conductivite": np.random.uniform(0, 1, n),
        "Opacite": np.random.uniform(0, 1, n),
        "Rigidite": np.random.uniform(1, 10, n),
        "Prix_Revente": np.random.uniform(0, 20, n),
        "Source": np.random.choice(sources, n),
        "Rapport_Collecte": [
            f"Lot de {cat.lower()} collecté, poids important."
            for cat in cat_labels
        ],
        "Categorie": cat_labels,
        "Categorie_enc": cat_enc,
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def labeled_data(sample_data):
    return sample_data.dropna(subset=["Categorie_enc"]).copy()


# =============================================================================
# ARTEFACTS ML
# =============================================================================

@pytest.fixture(scope="session")
def fitted_scaler(labeled_data):
    cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
    df = labeled_data[cols].fillna(labeled_data[cols].median())

    sc = StandardScaler()
    sc.fit(df)
    return sc


@pytest.fixture(scope="session")
def fitted_ohe(labeled_data):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(labeled_data[["Source"]])
    return ohe


@pytest.fixture(scope="session")
def fitted_le(labeled_data):
    le = LabelEncoder()
    le.fit(labeled_data["Categorie"])
    return le


@pytest.fixture(scope="session")
def fitted_tfidf(labeled_data):
    from src.module_4_NLP.text_classifier import preprocess_text

    texts = [preprocess_text(t) for t in labeled_data["Rapport_Collecte"].astype(str)]
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    vec.fit(texts)
    return vec


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def fitted_classifier(labeled_data, fitted_scaler, fitted_ohe, fitted_tfidf):
    import scipy.sparse as sp
    from src.module_4_NLP.text_classifier import preprocess_text

    cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
    df = labeled_data.fillna(labeled_data[cols].median())

    num = fitted_scaler.transform(df[cols])[:, :-1]
    cat = fitted_ohe.transform(df[["Source"]])
    txt = fitted_tfidf.transform([preprocess_text(t) for t in df["Rapport_Collecte"]])

    X = sp.hstack([sp.csr_matrix(num), sp.csr_matrix(cat), txt])
    y = df["Categorie_enc"].astype(int).values

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def fitted_regressor(labeled_data, fitted_scaler, fitted_ohe, fitted_tfidf):
    import scipy.sparse as sp
    from src.module_4_NLP.text_classifier import preprocess_text

    cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
    df = labeled_data.fillna(labeled_data[cols].median())

    num = fitted_scaler.transform(df[cols])[:, :-1]
    cat = fitted_ohe.transform(df[["Source"]])
    txt = fitted_tfidf.transform([preprocess_text(t) for t in df["Rapport_Collecte"]])

    X = sp.hstack([sp.csr_matrix(num), sp.csr_matrix(cat), txt])
    y = df["Prix_Revente"].values

    model = LinearRegression()
    model.fit(X, y)
    return model


# =============================================================================
# PYTEST CONFIG (FIX WARNINGS PROPREMENT)
# =============================================================================

def pytest_configure():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="matplotlib")
    warnings.filterwarnings("ignore", module="PIL")