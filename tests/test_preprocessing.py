# =============================================================================
# tests/test_preprocessing.py
# Tests du Module 1 : schéma, imputation, qualité post-nettoyage
# =============================================================================

import pytest
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder

from src.module_1_pretraitement.preprocessing import load_data


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def sample_data():
    """Dataset synthétique avec labels manquants simulés."""
    df = load_data(use_synthetic=True)

    rng = np.random.default_rng(42)
    mask = rng.random(len(df)) < 0.15
    df.loc[mask, "Categorie"] = np.nan

    return df


@pytest.fixture
def labeled_data(sample_data):
    """Dataset avec encodage des labels."""
    df = sample_data.copy()

    df = df.dropna(subset=["Categorie"])

    le = LabelEncoder()
    df["Categorie_enc"] = le.fit_transform(df["Categorie"])

    return df


# =============================================================================
# 1. TESTS DE SCHÉMA DES DONNÉES
# =============================================================================

class TestDataSchema:

    REQUIRED_COLS = [
        "Poids", "Volume", "Conductivite", "Opacite",
        "Rigidite", "Prix_Revente", "Source",
        "Rapport_Collecte", "Categorie",
    ]

    NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]

    def test_required_columns_present(self, sample_data):
        for col in self.REQUIRED_COLS:
            assert col in sample_data.columns

    def test_column_count(self, sample_data):
        assert sample_data.shape[1] >= 9

    def test_minimum_rows(self, sample_data):
        assert sample_data.shape[0] >= 100

    def test_numeric_columns_dtype(self, sample_data):
        for col in self.NUM_COLS:
            assert pd.api.types.is_numeric_dtype(sample_data[col])

    def test_text_column_dtype(self, sample_data):
        assert sample_data["Rapport_Collecte"].dtype == object

    def test_poids_missing_rate(self, sample_data):
        rate = sample_data["Poids"].isnull().mean()
        assert 0.02 <= rate <= 0.20

    def test_target_has_nulls(self, sample_data):
        assert sample_data["Categorie"].isnull().any()


# =============================================================================
# 2. TESTS DE CHARGEMENT
# =============================================================================

class TestLoadData:

    def test_load_synthetic(self):
        df = load_data(use_synthetic=True)
        assert isinstance(df, pd.DataFrame)

    def test_load_synthetic_has_rapport(self):
        df = load_data(use_synthetic=True)
        assert "Rapport_Collecte" in df.columns

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data(path="data/raw/fake.csv", use_synthetic=False)


# =============================================================================
# 3. TESTS D'IMPUTATION
# =============================================================================

class TestImputation:

    NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]

    def test_median_imputation_no_nan(self, sample_data):
        df = sample_data[self.NUM_COLS].copy()
        for col in self.NUM_COLS:
            df[col] = df[col].fillna(df[col].median())
        assert df.isnull().sum().sum() == 0

    def test_knn_imputation_no_nan(self, sample_data):
        df = sample_data[self.NUM_COLS].copy()
        knn = KNNImputer(n_neighbors=5)
        result = knn.fit_transform(df)
        assert np.isnan(result).sum() == 0

    def test_iterative_imputation_no_nan(self, sample_data):
        df = sample_data[self.NUM_COLS].copy()
        it = IterativeImputer(random_state=42, max_iter=5)
        result = it.fit_transform(df)
        assert np.isnan(result).sum() == 0

    def test_imputed_values_in_range(self, sample_data):
        df = sample_data[self.NUM_COLS].copy()
        for col in self.NUM_COLS:
            df[col] = df[col].fillna(df[col].median())

        poids_median = sample_data["Poids"].median()
        assert df["Poids"].max() <= poids_median * 15

    def test_imputation_preserves_non_null(self, sample_data):
        df = sample_data[self.NUM_COLS].copy()
        original = df["Conductivite"].dropna().values.copy()

        df["Conductivite"] = df["Conductivite"].fillna(df["Conductivite"].median())

        np.testing.assert_array_almost_equal(
            df.loc[sample_data["Conductivite"].notna(), "Conductivite"].values,
            original,
        )


# =============================================================================
# 4. TESTS OUTLIERS
# =============================================================================

class TestOutliers:

    def test_iqr_clipping_removes_extremes(self, sample_data):
        from src.module_1_pretraitement.preprocessing import handle_outliers

        df_clean = handle_outliers(sample_data.copy())

        for col in ["Poids", "Volume"]:
            if col not in df_clean.columns:
                continue

            for cat in sample_data["Categorie"].dropna().unique():
                mask_orig = sample_data["Categorie"] == cat
                mask_clean = df_clean["Categorie"] == cat

                group = sample_data.loc[mask_orig, col].dropna()

                Q1, Q3 = group.quantile(0.25), group.quantile(0.75)
                IQR = Q3 - Q1
                upper = Q3 + 1.5 * IQR
                lower = Q1 - 1.5 * IQR

                assert df_clean.loc[mask_clean, col].max() <= upper + 1e-9
                assert df_clean.loc[mask_clean, col].min() >= lower - 1e-9

    def test_clipping_preserves_shape(self, sample_data):
        from src.module_1_pretraitement.preprocessing import handle_outliers
        df_clean = handle_outliers(sample_data.copy())
        assert df_clean.shape[0] == sample_data.shape[0]


# =============================================================================
# 5. TEST SPLIT
# =============================================================================

class TestSplit:

    def test_split_ratios(self, labeled_data):
        from sklearn.model_selection import StratifiedShuffleSplit

        X = labeled_data.drop(columns=["Categorie", "Categorie_enc"], errors="ignore")
        y = labeled_data["Categorie_enc"].astype(int).values

        n = len(y)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        _, te_idx = next(sss.split(X, y))

        assert abs(len(te_idx) / n - 0.15) < 0.03

    def test_split_stratification(self, labeled_data):
        from sklearn.model_selection import StratifiedShuffleSplit

        X = labeled_data.drop(columns=["Categorie", "Categorie_enc"], errors="ignore")
        y = labeled_data["Categorie_enc"].astype(int).values

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(X, y))

        train_dist = np.bincount(y[tr_idx]) / len(tr_idx)
        test_dist = np.bincount(y[te_idx]) / len(te_idx)

        np.testing.assert_allclose(train_dist, test_dist, atol=0.10)

    def test_no_overlap_between_splits(self, labeled_data):
        from sklearn.model_selection import StratifiedShuffleSplit

        X = labeled_data.drop(columns=["Categorie", "Categorie_enc"], errors="ignore")
        y = labeled_data["Categorie_enc"].astype(int).values

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tv_idx, te_idx = next(sss.split(X, y))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
        X_tv = X.iloc[tv_idx]
        y_tv = y[tv_idx]

        tr_rel, va_rel = next(sss2.split(X_tv, y_tv))

        tr_abs = set(tv_idx[tr_rel])
        va_abs = set(tv_idx[va_rel])
        te_abs = set(te_idx)

        assert len(tr_abs & va_abs) == 0
        assert len(tr_abs & te_abs) == 0
        assert len(va_abs & te_abs) == 0


# =============================================================================
# 6. PIPELINE INTEGRATION
# =============================================================================

class TestPipelineIntegration:

    def test_run_preprocessing_synthetic(self, tmp_path, monkeypatch):
        import os
        from src.module_1_pretraitement.preprocessing import run_preprocessing

        monkeypatch.chdir(tmp_path)

        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        df = run_preprocessing(use_synthetic=True)

        assert df is not None
        assert "Categorie_enc" in df.columns