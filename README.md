# ♻️ Eco-Smart Classifier

Application intelligente de classification et d’estimation de valeur des déchets recyclables à l’aide du Machine Learning, NLP et MLOps.

---

# 📌 Objectif du Projet

Le projet **Eco-Smart Classifier** a pour objectif de construire un pipeline complet de Data Science capable de :

- classifier automatiquement des déchets,
- estimer leur prix de revente,
- analyser des descriptions textuelles,
- segmenter les matériaux via clustering,
- déployer le modèle dans une API et une application web.

---

# 🚀 Fonctionnalités

## ✅ Data Engineering
- Analyse exploratoire des données (EDA)
- Gestion des valeurs manquantes
- Détection et traitement des outliers
- Encodage des variables catégoriques
- Normalisation / standardisation

## ✅ Machine Learning Supervisé
- Classification des déchets
- Régression du prix de revente
- Feature Selection avec SHAP
- Hyperparameter tuning (GridSearchCV / Optuna)

## ✅ Machine Learning Non Supervisé
- Clustering avec KMeans
- Détermination du nombre optimal de clusters
- Visualisation PCA 2D

## ✅ NLP
- Nettoyage de texte
- Tokenization
- Suppression des stopwords
- TF-IDF
- Word2Vec
- FastText

## ✅ Pipeline Multimodal
Fusion :
- données numériques
- données textuelles

via :
- `ColumnTransformer`
- `hstack`

## ✅ MLOps
- Git + DVC
- MLflow Tracking
- CI/CD GitHub Actions
- Docker
- FastAPI
- Monitoring avec Evidently AI

---

# 🏗️ Architecture du Projet

```bash
Eco-Smart-Classifier/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── nlp/
│   ├── clustering/
│   ├── api/
│
├── tests/
│
├── mlruns/
│
├── dvc.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── PROMPTS.md
