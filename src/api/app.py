# =============================================================================
# API Eco-Smart Classifier - FastAPI (Version complète avec Monitoring)
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Modèles chargés globalement
models = {}

# =============================================================================
# MONITORING PROMETHEUS (Bonus 6)
# =============================================================================

try:
    from src.monitoring.prometheus_metrics import (
        start_metrics_server, track_prediction, track_error,
        track_active_request, get_metrics
    )
    PROMETHEUS_AVAILABLE = True
    start_metrics_server(8001)
    print("✓ Prometheus metrics server started on port 8001")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠ Prometheus metrics not available (module not found)")

    def track_prediction(categorie, latency, confidence): pass
    def track_error(error_type): pass
    def track_active_request(start): pass
    def get_metrics(): return b""


# =============================================================================
# LIFESPAN (chargement des modèles)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage"""
    print("=" * 60)
    print("CHARGEMENT DES MODÈLES")
    print("=" * 60)

    try:
        models["scaler"] = joblib.load(os.path.join(PROCESSED_DIR, "scaler.pkl"))
        models["ohe"] = joblib.load(os.path.join(PROCESSED_DIR, "onehot_encoder.pkl"))
        models["le"] = joblib.load(os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
        print("  ✓ Preprocessing models loaded")

        classifier_path = os.path.join(MODELS_DIR, "multimodal_classifier.pkl")
        if os.path.exists(classifier_path):
            models["classifier"] = joblib.load(classifier_path)
            print("  ✓ Multimodal classifier loaded")
        else:
            models["classifier"] = None
            print("  ✗ Multimodal classifier not found")

        vectorizer_paths = [
            os.path.join(MODELS_DIR, "nlp", "tfidf_vectorizer.pkl"),
            os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
        ]
        vectorizer_found = False
        for vectorizer_path in vectorizer_paths:
            if os.path.exists(vectorizer_path):
                models["vectorizer"] = joblib.load(vectorizer_path)
                print(f"  ✓ TF-IDF vectorizer loaded from {vectorizer_path}")
                vectorizer_found = True
                break
        if not vectorizer_found:
            models["vectorizer"] = None
            print("  ✗ TF-IDF vectorizer not found")

        regressor_path = os.path.join(MODELS_DIR, "multimodal_regressor.pkl")
        if os.path.exists(regressor_path):
            models["regressor"] = joblib.load(regressor_path)
            print("  ✓ Multimodal regressor loaded")
        else:
            models["regressor"] = None
            print("  ⚠ Multimodal regressor not found (price estimation disabled)")

        feature_names_path = os.path.join(MODELS_DIR, "num_feature_names.pkl")
        if os.path.exists(feature_names_path):
            models["feature_names"] = joblib.load(feature_names_path)
            print("  ✓ Feature names loaded")

        print("=" * 60)
    except Exception as e:
        print(f"  ✗ Error loading models: {e}")

    yield
    print("Fermeture de l'API...")


# =============================================================================
# CRÉATION DE L'APPLICATION
# =============================================================================

app = FastAPI(
    title="Eco-Smart Classifier API",
    description="""
    ## API de classification et estimation de déchets

    Cette API permet de :
    - **Classifier** un déchet dans une catégorie (Papier, Plastique, Verre, Métal)
    - **Estimer** son prix de revente
    - Utiliser une description textuelle pour améliorer la prédiction

    ### Catégories disponibles
    - Papier
    - Plastique
    - Verre
    - Métal
    """,
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# MODÈLES DE DONNÉES (pydantic v2)
# =============================================================================

class WasteInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Poids": 50.0,
                "Volume": 100.0,
                "Conductivite": 0.5,
                "Opacite": 0.8,
                "Rigidite": 5.0,
                "Source": "Usine_A",
                "Rapport_Collecte": "Lot de plastique récupéré. Poids moyen, volume important. Matériau recyclable."
            }
        }
    )

    Poids: float = Field(..., ge=0, le=1000, description="Poids du déchet en kg")
    Volume: float = Field(..., ge=0, le=10000, description="Volume du déchet en litres")
    Conductivite: float = Field(..., ge=0, le=1, description="Conductivité électrique (0-1)")
    Opacite: float = Field(..., ge=0, le=1, description="Opacité du matériau (0-1)")
    Rigidite: float = Field(..., ge=1, le=10, description="Rigidité (1=très souple, 10=très rigide)")
    Source: str = Field(..., description="Source du déchet (Usine_A, Usine_B, Centre_Tri, Collecte_Citoyenne)")
    Rapport_Collecte: str = Field(..., min_length=3, description="Description textuelle du déchet")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "categorie_predite": "Plastique",
                "prix_estime": 4.50,
                "probabilites": {
                    "Métal": 0.05,
                    "Papier": 0.02,
                    "Plastique": 0.90,
                    "Verre": 0.03
                },
                "confidence": 0.90
            }
        }
    )

    categorie_predite: str = Field(..., description="Catégorie prédite")
    prix_estime: Optional[float] = Field(None, description="Prix de revente estimé (€)")
    probabilites: Optional[Dict[str, float]] = Field(None, description="Probabilités par catégorie")
    confidence: Optional[float] = Field(None, description="Niveau de confiance (0-1)")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def preprocess_text(text: str) -> str:
    """Prétraitement du texte (identique au Module 4)"""
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        stop_words = set(stopwords.words('french'))
        lemmatizer = WordNetLemmatizer()
    except Exception:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words = set(stopwords.words('french'))
        lemmatizer = WordNetLemmatizer()

    stop_words.update(["lot", "kg", "l", "matériau", "déchet"])

    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëïîôùûüÿ\s]', '', text)
    tokens = word_tokenize(text, language='french')
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 2]
    return " ".join(tokens)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eco-Smart Classifier API</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                   max-width: 800px; margin: 50px auto; padding: 20px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; }
            .container { background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2); color: #333; }
            h1 { color: #667eea; }
            .endpoint { background: #f0f0f0; padding: 10px; border-radius: 5px;
                       font-family: monospace; margin: 10px 0; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
            a { color: #667eea; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>♻️ Eco-Smart Classifier API</h1>
            <p>API de classification automatique des déchets et estimation du prix de revente.</p>
            <h2>📡 Endpoints disponibles</h2>
            <div class="endpoint"><strong>POST /predict</strong> - Prédire la catégorie et le prix</div>
            <div class="endpoint"><strong>GET /health</strong> - Vérifier l'état de l'API</div>
            <div class="endpoint"><strong>GET /metrics</strong> - Métriques Prometheus</div>
            <div class="endpoint"><strong>GET /docs</strong> - Documentation interactive (Swagger)</div>
            <div class="endpoint"><strong>GET /redoc</strong> - Documentation alternative (ReDoc)</div>
            <p style="margin-top: 30px;">
                <a href="/docs">📖 Voir la documentation Swagger</a> |
                <a href="/redoc">📚 Voir ReDoc</a>
            </p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "multimodal_available": models.get("classifier") is not None,
        "prometheus_available": PROMETHEUS_AVAILABLE
    }


@app.get("/metrics")
async def metrics():
    """Endpoint pour Prometheus (Bonus 6)"""
    return Response(content=get_metrics(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse)
async def predict(waste: WasteInput):
    """Prédiction de la catégorie et du prix de revente"""
    start_time = time.time()
    track_active_request(start=True)

    try:
        if models.get("classifier") is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        if models.get("vectorizer") is None:
            raise HTTPException(status_code=503, detail="Vectoriseur non disponible")

        # 1. Features numériques
        num_data = pd.DataFrame([[
            waste.Poids, waste.Volume, waste.Conductivite,
            waste.Opacite, waste.Rigidite, 0.0
        ]], columns=["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"])
        num_scaled = models["scaler"].transform(num_data)[:, :-1]  # exclure Prix_Revente

        # 2. One-hot encoding Source
        src_df = pd.DataFrame([[waste.Source]], columns=["Source"])
        src_encoded = models["ohe"].transform(src_df)
        src_array = src_encoded.toarray() if hasattr(src_encoded, "toarray") else src_encoded

        # 3. Texte vectorisé
        text_clean = preprocess_text(waste.Rapport_Collecte)
        text_vec = models["vectorizer"].transform([text_clean]).toarray()

        # 4. Fusion
        X_final = np.hstack([num_scaled, src_array, text_vec])

        # 5. Prédiction catégorie
        cat_enc = models["classifier"].predict(X_final)[0]
        categorie = models["le"].inverse_transform([cat_enc])[0]

        # 6. Probabilités
        probabilites = None
        confidence = None
        if hasattr(models["classifier"], "predict_proba"):
            probas = models["classifier"].predict_proba(X_final)[0]
            classes = models["le"].classes_
            probabilites = {cls: float(prob) for cls, prob in zip(classes, probas)}
            confidence = float(max(probas))

        # 7. Prix estimé
        prix = None
        if models.get("regressor") is not None:
            try:
                prix_val = models["regressor"].predict(X_final)[0]
                prix = max(0, round(float(prix_val), 2))
            except Exception:
                prix = None

        latency = time.time() - start_time
        track_prediction(categorie, latency, confidence or 0.95)

        return PredictionResponse(
            categorie_predite=categorie,
            prix_estime=prix,
            probabilites=probabilites,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        track_error(type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        track_active_request(start=False)

# =============================================================================
# MODÈLE POUR NLP-ONLY (sans validation stricte)
# =============================================================================

class NLPInput(BaseModel):
    """Entrée pour l'endpoint NLP (seul le texte compte)"""
    Rapport_Collecte: str = Field(..., min_length=3, description="Description textuelle du déchet")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Rapport_Collecte": "Bouteille en verre transparent, épaisse"
            }
        }


# =============================================================================
# MODÈLE POUR NLP-ONLY (une seule définition)
# =============================================================================

class NLPInput(BaseModel):
    """Entrée pour l'endpoint NLP (seul le texte compte)"""
    Rapport_Collecte: str = Field(..., min_length=1, description="Description textuelle du déchet")


# =============================================================================
# ENDPOINT NLP-ONLY (CORRIGÉ)
# =============================================================================

@app.post("/predict_nlp")
async def predict_nlp(item: NLPInput):
    """
    Prédiction basée UNIQUEMENT sur le texte (NLP pur)
    """
    try:
        from src.module_4_NLP.text_classifier import preprocess_text
        import joblib
        
        nlp_classifier_path = os.path.join(MODELS_DIR, "nlp", "best_nlp_classifier.pkl")
        nlp_vectorizer_path = os.path.join(MODELS_DIR, "nlp", "tfidf_vectorizer.pkl")
        
        if not os.path.exists(nlp_classifier_path):
            raise HTTPException(status_code=503, detail="Modèle NLP non disponible")
        
        nlp_classifier = joblib.load(nlp_classifier_path)
        nlp_vectorizer = joblib.load(nlp_vectorizer_path)
        
        text_clean = preprocess_text(item.Rapport_Collecte)
        text_vec = nlp_vectorizer.transform([text_clean]).toarray()
        
        cat_enc = nlp_classifier.predict(text_vec)[0]
        categorie = models["le"].inverse_transform([cat_enc])[0]
        
        # Calculer une confiance approximative (car LinearSVC n'a pas predict_proba)
        confidence = 0.85  # valeur par défaut
        
        if hasattr(nlp_classifier, "decision_function"):
            try:
                decision = nlp_classifier.decision_function(text_vec)
                # Si la valeur est élevée (loin de 0), la confiance est bonne
                if len(decision.shape) > 1:
                    max_decision = np.max(np.abs(decision[0]))
                else:
                    max_decision = np.abs(decision[0])
                confidence = min(0.95, 0.6 + max_decision / 10)
            except:
                pass
        
        # Seuil de confiance
        if confidence < 0.5:
            return PredictionResponse(
                categorie_predite="❓ Texte non reconnu comme déchet",
                prix_estime=None,
                probabilites=None,
                confidence=confidence
            )
        
        return PredictionResponse(
            categorie_predite=categorie,
            prix_estime=None,
            probabilites=None,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)