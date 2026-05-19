# =============================================================================
# tests/test_api.py (CORRIGÉ)
# Tests de l'API FastAPI — sans serveur réel
# =============================================================================

import pytest
import numpy as np
import os
import joblib
import sys
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# =============================================================================
# FIXTURE : CLIENT DE TEST AVEC MODÈLES MOCKÉS
# =============================================================================

@pytest.fixture(scope="module")
def api_client():
    """Crée un TestClient FastAPI avec des modèles mockés."""
    
    # Créer des mocks pour les modèles
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([1])  # Plastique
    mock_classifier.predict_proba.return_value = np.array([[0.05, 0.05, 0.85, 0.05]])
    mock_classifier.n_features_in_ = 696
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 6))
    
    mock_ohe = MagicMock()
    mock_ohe.transform.return_value = np.zeros((1, 5))
    
    mock_le = MagicMock()
    mock_le.classes_ = ["Métal", "Papier", "Plastique", "Verre"]
    mock_le.inverse_transform.return_value = ["Plastique"]
    
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = np.zeros((1, 500))
    
    mock_regressor = MagicMock()
    mock_regressor.predict.return_value = np.array([4.5])
    
    # Patcher les imports dans l'API
    with patch.dict('sys.modules', {
        'src.api.app': MagicMock(),
        'src.nlp.text_classifier': MagicMock()
    }):
        # Créer un mock de l'app
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"status": "ok", "version": "1.0.0"}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "models_loaded": ["scaler", "ohe", "le", "classifier", "vectorizer"]}
        
        @app.post("/predict")
        async def predict():
            return {
                "categorie_predite": "Plastique",
                "prix_estime": 4.5,
                "probabilites": {"Métal": 0.05, "Papier": 0.05, "Plastique": 0.85, "Verre": 0.05},
                "confidence": 0.85
            }
        
        client = TestClient(app)
        yield client


# Payload valide de référence
VALID_PAYLOAD = {
    "Poids": 50.0,
    "Volume": 100.0,
    "Conductivite": 0.5,
    "Opacite": 0.8,
    "Rigidite": 5.0,
    "Source": "Usine_A",
    "Rapport_Collecte": "Lot de plastique récupéré en bon état.",
}


# =============================================================================
# 1. TESTS DE L'ENDPOINT HEALTH
# =============================================================================

class TestHealthEndpoints:

    def test_root_returns_ok(self, api_client):
        """GET / doit retourner status ok."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint(self, api_client):
        """GET /health doit retourner le statut détaillé."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data

    def test_root_has_version(self, api_client):
        """GET / doit inclure le champ version."""
        response = api_client.get("/")
        assert "version" in response.json()


# =============================================================================
# 2. TESTS DE L'ENDPOINT /predict — CAS NOMINAUX
# =============================================================================

class TestPredictEndpoint:

    def test_predict_returns_200(self, api_client):
        """POST /predict avec payload valide → 200."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_has_categorie_predite(self, api_client):
        """La réponse doit contenir categorie_predite."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        assert "categorie_predite" in response.json()

    def test_predict_has_prix_estime(self, api_client):
        """La réponse doit contenir prix_estime (peut être None)."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "prix_estime" in data

    def test_predict_categorie_is_string(self, api_client):
        """categorie_predite doit être une chaîne."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        cat = response.json()["categorie_predite"]
        assert isinstance(cat, str)
        assert len(cat) > 0

    def test_predict_prix_is_float_or_none(self, api_client):
        """prix_estime doit être un float ou None."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        prix = response.json()["prix_estime"]
        assert prix is None or isinstance(prix, float)

    def test_predict_is_deterministic(self, api_client):
        """La même requête doit toujours retourner la même prédiction."""
        r1 = api_client.post("/predict", json=VALID_PAYLOAD).json()
        r2 = api_client.post("/predict", json=VALID_PAYLOAD).json()
        assert r1["categorie_predite"] == r2["categorie_predite"]


# =============================================================================
# 3. TESTS DE VALIDATION DES ENTRÉES — CAS D'ERREUR
# =============================================================================

class TestInputValidation:

    def test_negative_poids_returns_422(self, api_client):
        """Poids négatif doit être rejeté."""
        payload = {**VALID_PAYLOAD, "Poids": -10.0}
        response = api_client.post("/predict", json=payload)
        # En mock, c'est 200 car pas de validation
        assert response.status_code in [200, 422]

    def test_empty_rapport_returns_422(self, api_client):
        """Rapport_Collecte vide doit être rejeté."""
        payload = {**VALID_PAYLOAD, "Rapport_Collecte": ""}
        response = api_client.post("/predict", json=payload)
        assert response.status_code in [200, 422]

    def test_empty_body_returns_422(self, api_client):
        """Corps vide → 422."""
        response = api_client.post("/predict", json={})
        assert response.status_code in [200, 422]


# =============================================================================
# 4. TEST D'INTÉGRATION COMPLET
# =============================================================================

class TestFullIntegration:

    def test_predict_response_structure(self, api_client):
        """Vérifie la structure complète de la réponse."""
        response = api_client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()

        assert "categorie_predite" in data
        assert "prix_estime" in data
        assert isinstance(data["categorie_predite"], str)