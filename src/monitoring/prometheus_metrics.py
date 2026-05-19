# =============================================================================
# BONUS 6 — MONITORING PROMETHEUS + GRAFANA
# =============================================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest, start_http_server, REGISTRY
import time
import threading
import random

# =============================================================================
# DÉFINITION DES MÉTRIQUES
# =============================================================================

# Compteur de prédictions par catégorie
predictions_total = Counter(
    'eco_smart_predictions_total',
    'Total number of predictions',
    ['categorie']
)

# Latence des prédictions
prediction_latency = Histogram(
    'eco_smart_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Confiance du modèle
model_confidence = Gauge(
    'eco_smart_model_confidence',
    'Current model confidence score'
)

# Requêtes actives
active_requests = Gauge(
    'eco_smart_active_requests',
    'Number of active API requests'
)

# Distribution des catégories prédites
category_distribution = Counter(
    'eco_smart_category_distribution',
    'Distribution of predicted categories',
    ['categorie']
)

# Erreurs API
api_errors = Counter(
    'eco_smart_api_errors',
    'Number of API errors',
    ['error_type']
)

# Drift des données
data_drift_score = Gauge(
    'eco_smart_data_drift_score',
    'Data drift score (Jensen-Shannon divergence)'
)

# CPU et mémoire (optionnel)
import psutil
cpu_usage = Gauge('eco_smart_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('eco_smart_memory_usage_percent', 'Memory usage percentage')


# =============================================================================
# SERVEUR PROMETHEUS
# =============================================================================

def start_metrics_server(port: int = 8001):
    """Démarre le serveur Prometheus sur le port spécifié"""
    start_http_server(port)
    print(f"✅ Prometheus metrics server started on port {port}")
    
    # Démarrer le monitoring système en arrière-plan
    def monitor_system():
        while True:
            cpu_usage.set(psutil.cpu_percent())
            memory_usage.set(psutil.virtual_memory().percent)
            time.sleep(5)
    
    thread = threading.Thread(target=monitor_system, daemon=True)
    thread.start()


def get_metrics():
    """Retourne les métriques au format Prometheus"""
    return generate_latest(REGISTRY)


# =============================================================================
# FONCTIONS D'ENREGISTREMENT
# =============================================================================

def track_prediction(categorie: str, latency: float, confidence: float):
    """Enregistre les métriques d'une prédiction"""
    predictions_total.labels(categorie=categorie).inc()
    category_distribution.labels(categorie=categorie).inc()
    prediction_latency.observe(latency)
    model_confidence.set(confidence)


def track_error(error_type: str):
    """Enregistre une erreur API"""
    api_errors.labels(error_type=error_type).inc()


def track_data_drift(score: float):
    """Enregistre le score de drift des données"""
    data_drift_score.set(score)


def track_active_request(start: bool = True):
    """Enregistre le début/fin d'une requête"""
    if start:
        active_requests.inc()
    else:
        active_requests.dec()


# =============================================================================
# DASHBOARD GRAFANA (JSON CONFIG)
# =============================================================================

GRAFANA_DASHBOARD_JSON = {
    "title": "Eco-Smart Classifier Monitoring",
    "uid": "eco-smart-monitoring",
    "panels": [
        {
            "title": "Predictions per Category",
            "type": "bargauge",
            "targets": [{"expr": 'sum(eco_smart_predictions_total) by (categorie)'}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        },
        {
            "title": "Prediction Latency (ms)",
            "type": "heatmap",
            "targets": [{"expr": "eco_smart_prediction_latency_seconds_bucket"}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        },
        {
            "title": "Model Confidence",
            "type": "gauge",
            "targets": [{"expr": "eco_smart_model_confidence"}],
            "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8}
        },
        {
            "title": "Active Requests",
            "type": "stat",
            "targets": [{"expr": "eco_smart_active_requests"}],
            "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8}
        },
        {
            "title": "API Errors",
            "type": "piechart",
            "targets": [{"expr": 'sum(eco_smart_api_errors) by (error_type)'}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        },
        {
            "title": "CPU Usage",
            "type": "timeseries",
            "targets": [{"expr": "eco_smart_cpu_usage_percent"}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
        },
        {
            "title": "Memory Usage",
            "type": "timeseries",
            "targets": [{"expr": "eco_smart_memory_usage_percent"}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
        }
    ]
}


# =============================================================================
# FICHIER prometheus.yml (à la racine)
# =============================================================================

PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'eco-smart-api'
    static_configs:
      - targets: ['localhost:8001']
  - job_name: 'eco-smart-system'
    static_configs:
      - targets: ['localhost:8001']
"""