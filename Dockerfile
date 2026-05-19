FROM python:3.11-slim

WORKDIR /app

# =========================
# Dépendances système
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Installer dépendances Python
# =========================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# Installer gdown explicitement (AJOUTEZ CECI)
# =========================
RUN pip install gdown

# =========================
# Télécharger ressources NLTK
# =========================
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# =========================
# Copier le code source
# =========================
COPY src/ ./src/

# =========================
# Copier le script de téléchargement
# =========================
COPY download_models.py .

# =========================
# Télécharger les modèles depuis Google Drive
# =========================
RUN python download_models.py

# =========================
# Créer dossiers nécessaires
# =========================
RUN mkdir -p reports/figures logs

# =========================
# Créer __init__.py
# =========================
RUN touch src/__init__.py \
    src/api/__init__.py \
    src/application/__init__.py \
    src/module_1_pretraitement/__init__.py \
    src/module_2_class_reg/__init__.py \
    src/module_3_clustering/__init__.py \
    src/module_4_NLP/__init__.py \
    src/module_5_multimodal/__init__.py \
    src/monitoring/__init__.py

# =========================
# Exposer ports
# =========================
EXPOSE 8000 8501

# =========================
# Lancer FastAPI + Streamlit
# =========================
CMD sh -c "\
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & \
streamlit run src/application/app_web.py --server.address 0.0.0.0 --server.port 8501"