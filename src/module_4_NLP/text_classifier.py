# =============================================================================
# MODULE 4 — NLP : EXTRACTION DE CARACTÉRISTIQUES TEXTUELLES
# Projet : Eco-Smart Classifier
# Compatible avec TP7 : Extraction d'attributs (contamination, état, type)
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess

import mlflow
import mlflow.sklearn

# =============================================================================
# PATHS
# =============================================================================
CLEAN_DATA_PATH = "data/processed/clean_data.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"
NLP_MODELS_DIR = "models/nlp"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(NLP_MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Téléchargement des ressources NLTK
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

# =============================================================================
# STOPWORDS ET LEMMATIZER
# =============================================================================
_STOP_WORDS = set(stopwords.words("french"))
_STOP_WORDS.update(["lot", "kg", "matériau", "déchet", "collecte",
                     "récupéré", "récupérée", "type", "l", "ml", "cm", "gr",
                     "poids", "volume", "conductivite", "opacite", "rigidite"])
_LEMMATIZER = WordNetLemmatizer()


# =============================================================================
# 1. PRÉTRAITEMENT TEXTUEL
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Nettoie, tokenise, supprime les stopwords et lemmatise un texte.
    
    Note: Garde les chiffres pour les unités (kg, l, etc.)
    """
    if not isinstance(text, str):
        return ""

    # 1. Minuscules
    text = text.lower()

    # 2. Suppression ponctuation (MAIS GARDE LES CHIFFRES ET LE POINT)
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = re.sub(r"[^a-zàâçéèêëïîôùûüÿ0-9\.\s]", " ", text)

    # 3. Normalisation des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenisation française
    try:
        tokens = word_tokenize(text, language="french")
    except Exception:
        tokens = text.split()

    # 5. Filtrage stopwords + longueur + lemmatisation
    tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOP_WORDS and len(token) > 2
    ]

    return " ".join(tokens)


# =============================================================================
# 2. EXTRACTION D'ATTRIBUTS (TP7 Partie 2.2)
# =============================================================================

def extract_contamination(text: str) -> int:
    """Extrait l'indicateur de contamination du texte."""
    text_lower = text.lower()
    contamination_keywords = [
        "contamination", "humidité", "traces", "rouille", "corrosion",
        "moisissure", "salissure", "tache", "dégradation", "usure",
        "encrassement", "dépôt", "résidu", "pollution", "détérioration",
        "crasse", "oxydation", "verdigris", "souillé", "sale"
    ]
    for keyword in contamination_keywords:
        if keyword in text_lower:
            return 1
    return 0


def extract_material_state(text: str) -> str:
    """Extrait l'état du matériau du texte."""
    text_lower = text.lower()
    
    # État Brisé
    broken_keywords = ["cassé", "brisé", "fissuré", "endommagé", "dégradé",
                       "abîmé", "délabré", "déchiré", "percé", "fendu",
                       "éclaté", "fracturé", "disloqué", "usé"]
    for keyword in broken_keywords:
        if keyword in text_lower:
            return "Brisé"
    
    # État Neuf
    new_keywords = ["neuf", "nouveau", "original", "intact", "impeccable",
                    "pristine", "inusité", "jamais utilisé", "emballage"]
    for keyword in new_keywords:
        if keyword in text_lower:
            return "Neuf"
    
    # État Bon
    good_keywords = ["bon", "très bon", "excellent", "parfait", "propre",
                     "léger", "légère", "mineure", "à peine usé"]
    for keyword in good_keywords:
        if keyword in text_lower:
            return "Bon"
    
    return "Moyen"


def extract_material_type(text: str) -> str:
    """Extrait le type de matériau du texte."""
    text_lower = text.lower()
    
    material_keywords = {
        "Plastique": ["plastique", "pvc", "polypropylène", "polyéthylène", "pet", "pebd"],
        "Métal": ["métal", "aluminium", "acier", "cuivre", "fer", "inox", "laiton", "bronze"],
        "Verre": ["verre", "cristal", "vitreux", "bouteille", "contenant"],
        "Papier": ["papier", "carton", "cardboard", "kraft", "journal", "magazine"],
        "Bois": ["bois", "wooden", "timber", "palette"],
        "Textile": ["textile", "tissu", "coton", "laine", "fibre", "polyester"],
        "Caoutchouc": ["caoutchouc", "rubber", "élastique", "pneu"]
    }
    
    for material_type, keywords in material_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return material_type
    return "Inconnu"


def extract_text_features(text: str) -> dict:
    """Extraction complète des attributs du texte."""
    return {
        "contamination": extract_contamination(text),
        "etat_materiau": extract_material_state(text),
        "type_materiau": extract_material_type(text)
    }


# =============================================================================
# 3. VECTORISATION
# =============================================================================

def text_to_embedding(text: str, model, vector_size: int = 100) -> np.ndarray:
    """Convertit un texte en vecteur moyen Word2Vec ou FastText."""
    tokens = simple_preprocess(text)
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


# =============================================================================
# 4. PIPELINE D'ENTRAÎNEMENT NLP COMPLET (avec MLflow)
# =============================================================================

def run_nlp_pipeline(data_path: str = CLEAN_DATA_PATH,
                     experiment_name: str = "Eco-Smart NLP Pipeline") -> dict:
    """
    Exécute le pipeline NLP complet avec tracking MLflow.
    """
    
    # =========================================================================
    # SETUP MLflow
    # =========================================================================
    mlflow.set_experiment(experiment_name)
    
    print("=" * 60)
    print("MODULE 4 — PIPELINE NLP (avec MLflow)")
    print("=" * 60)

    # --- Chargement ---
    print("\n1. CHARGEMENT DES DONNÉES")
    print("-" * 40)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Categorie_enc"]).copy()
    print(f"Lignes avec catégorie : {df.shape[0]}")

    texts = df["Rapport_Collecte"].astype(str).values
    labels = df["Categorie_enc"].astype(int).values

    # --- Extraction des attributs textuels ---
    print("\n2. EXTRACTION DES ATTRIBUTS TEXTUELS")
    print("-" * 40)
    
    extracted_features = df["Rapport_Collecte"].apply(extract_text_features)
    df["Contamination"] = extracted_features.apply(lambda x: x["contamination"])
    df["Etat_Materiau"] = extracted_features.apply(lambda x: x["etat_materiau"])
    df["Type_Materiau"] = extracted_features.apply(lambda x: x["type_materiau"])
    
    print(f"✓ Contamination détectée : {df['Contamination'].sum()} cas")
    print(f"✓ Distribution État : {df['Etat_Materiau'].value_counts().to_dict()}")
    
    # Sauvegarde
    df.to_csv(os.path.join(PROCESSED_DIR, "clean_data_with_text_features.csv"), index=False)
    print(f"✓ Données enrichies sauvegardées")

    # --- Prétraitement ---
    print("\n3. PRÉTRAITEMENT DES TEXTES")
    print("-" * 40)
    with mlflow.start_run(run_name="Preprocessing"):
        processed = [preprocess_text(t) for t in texts]
        print(f"Exemple original: {texts[0][:60]}...")
        print(f"Après prétraitement: {processed[0][:60]}...")
        mlflow.log_param("preprocessing_method", "lemmatization")

    # --- Split ---
    print("\n4. SPLIT TRAIN/TEST")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    results = {}
    classifiers = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    def _evaluate(clf, X_tr, X_te, y_tr, y_te, name, method_name):
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="weighted")
        print(f"    {name:25s} — Acc: {acc:.4f} | F1: {f1:.4f}")
        return acc, f1

    # ------------------------------------------------------------------ BoW
    print("\n5. BAG OF WORDS")
    print("-" * 40)
    with mlflow.start_run(run_name="BoW"):
        bow_vec = CountVectorizer(max_features=5000)
        X_tr_bow = bow_vec.fit_transform(X_train)
        X_te_bow = bow_vec.transform(X_test)
        mlflow.log_param("bow_max_features", 5000)
        
        for name, clf in classifiers.items():
            acc, f1 = _evaluate(clf, X_tr_bow, X_te_bow, y_train, y_test, name, "BoW")
            results[f"BoW_{name}"] = (acc, f1)
            mlflow.log_metric(f"BoW_{name}_accuracy", acc)
            mlflow.log_metric(f"BoW_{name}_f1", f1)

    # ---------------------------------------------------------------- TF-IDF
    print("\n6. TF-IDF (uni+bigrammes)")
    print("-" * 40)
    with mlflow.start_run(run_name="TFIDF"):
        tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tr_tfidf = tfidf_vec.fit_transform(X_train)
        X_te_tfidf = tfidf_vec.transform(X_test)
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", "(1,2)")
        
        for name, clf in classifiers.items():
            acc, f1 = _evaluate(clf, X_tr_tfidf, X_te_tfidf, y_train, y_test, name, "TFIDF")
            results[f"TFIDF_{name}"] = (acc, f1)
            mlflow.log_metric(f"TFIDF_{name}_accuracy", acc)
            mlflow.log_metric(f"TFIDF_{name}_f1", f1)

    # --------------------------------------------------------------- Word2Vec
    print("\n7. WORD2VEC")
    print("-" * 40)
    with mlflow.start_run(run_name="Word2Vec"):
        train_sentences = [simple_preprocess(s) for s in X_train]
        w2v = Word2Vec(sentences=train_sentences, vector_size=100,
                       window=5, min_count=2, workers=4, epochs=10)
        X_tr_w2v = np.array([text_to_embedding(t, w2v) for t in X_train])
        X_te_w2v = np.array([text_to_embedding(t, w2v) for t in X_test])
        sc_w2v = StandardScaler()
        X_tr_w2v = sc_w2v.fit_transform(X_tr_w2v)
        X_te_w2v = sc_w2v.transform(X_te_w2v)
        
        mlflow.log_param("word2vec_vector_size", 100)
        
        for name, clf in classifiers.items():
            if name == "MultinomialNB":
                continue
            acc, f1 = _evaluate(clf, X_tr_w2v, X_te_w2v, y_train, y_test, name, "W2V")
            results[f"W2V_{name}"] = (acc, f1)
            mlflow.log_metric(f"W2V_{name}_accuracy", acc)
            mlflow.log_metric(f"W2V_{name}_f1", f1)

    # --------------------------------------------------------------- FastText
    print("\n8. FASTTEXT")
    print("-" * 40)
    with mlflow.start_run(run_name="FastText"):
        ft = FastText(sentences=train_sentences, vector_size=100,
                      window=5, min_count=2, workers=4, epochs=10)
        X_tr_ft = np.array([text_to_embedding(t, ft) for t in X_train])
        X_te_ft = np.array([text_to_embedding(t, ft) for t in X_test])
        sc_ft = StandardScaler()
        X_tr_ft = sc_ft.fit_transform(X_tr_ft)
        X_te_ft = sc_ft.transform(X_te_ft)
        
        mlflow.log_param("fasttext_vector_size", 100)
        
        for name, clf in classifiers.items():
            if name == "MultinomialNB":
                continue
            acc, f1 = _evaluate(clf, X_tr_ft, X_te_ft, y_train, y_test, name, "FastText")
            results[f"FastText_{name}"] = (acc, f1)
            mlflow.log_metric(f"FastText_{name}_accuracy", acc)
            mlflow.log_metric(f"FastText_{name}_f1", f1)

    # --- Comparaison ---
    print("\n9. COMPARAISON DES MÉTHODES")
    print("-" * 40)
    df_results = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in results.items()],
        columns=["Méthode", "Accuracy", "F1_weighted"],
    ).sort_values("F1_weighted", ascending=False)

    print("\nCLASSEMENT DES MÉTHODES (F1 weighted):")
    print(df_results.to_string(index=False))

    # Visualisation
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="F1_weighted", y="Méthode", palette="viridis")
    plt.title("Comparaison NLP — F1 weighted")
    plt.xlabel("F1 score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig11_nlp_comparison.png"), dpi=120)
    plt.close()
    print("\n  ✓ fig11_nlp_comparison.png sauvegardé")

    # --- Meilleur modèle ---
    print("\n10. MEILLEUR MODÈLE")
    print("-" * 40)
    best_clf = LinearSVC(random_state=42, max_iter=2000)
    best_clf.fit(X_tr_tfidf, y_train)
    y_pred_best = best_clf.predict(X_te_tfidf)
    
    best_acc = accuracy_score(y_test, y_pred_best)
    best_f1 = f1_score(y_test, y_pred_best, average="weighted")
    print(f"Modèle final: TF-IDF + LinearSVC")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1-score: {best_f1:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de Confusion - TF-IDF + LinearSVC")
    plt.ylabel("Vrai label")
    plt.xlabel("Label prédit")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig12_nlp_confusion_matrix.png"), dpi=120)
    plt.close()
    print("  ✓ fig12_nlp_confusion_matrix.png sauvegardé")

    # --- Sauvegarde ---
    print("\n11. SAUVEGARDE DES MODÈLES")
    print("-" * 40)
    # Vectoriseur TF-IDF
    joblib.dump(tfidf_vec, os.path.join(NLP_MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(tfidf_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    # Classifieur
    joblib.dump(best_clf, os.path.join(NLP_MODELS_DIR, "best_nlp_classifier.pkl"))
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "best_nlp_classifier.pkl"))
    # Word2Vec et FastText
    joblib.dump(w2v, os.path.join(NLP_MODELS_DIR, "word2vec_model.pkl"))
    joblib.dump(ft, os.path.join(NLP_MODELS_DIR, "fasttext_model.pkl"))
    
    print("  ✓ tfidf_vectorizer.pkl")
    print("  ✓ best_nlp_classifier.pkl")
    print("  ✓ word2vec_model.pkl")
    print("  ✓ fasttext_model.pkl")
    
    # Sauvegarde des résultats JSON
    df_results.to_csv(os.path.join(FIGURES_DIR, "nlp_comparison_results.csv"), index=False)
    print("  ✓ nlp_comparison_results.csv sauvegardé")

    print("\n" + "=" * 60)
    print("MODULE 4 TERMINÉ ✓")
    print("=" * 60)

    return results

# =============================================================================
# BONUS 5 — SENTENCE TRANSFORMERS (CamemBERT)
# =============================================================================

def run_camembert_pipeline(data_path: str = CLEAN_DATA_PATH) -> dict:
    """
    Bonus : Utilisation de CamemBERT (modèle transformer français)
    Alternative avancée au TF-IDF pour la classification textuelle
    """
    print("\n" + "=" * 60)
    print("BONUS 5 — CAMEMBERT (Sentence Transformers)")
    print("=" * 60)
    
    from sentence_transformers import SentenceTransformer
    import mlflow
    
    # 1. Chargement
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Categorie_enc"]).copy()
    print(f"Lignes avec catégorie : {df.shape[0]}")
    
    texts = df["Rapport_Collecte"].astype(str).values
    labels = df["Categorie_enc"].astype(int).values
    
    # 2. Modèle CamemBERT (français)
    print("\nChargement du modèle CamemBERT...")
    model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    
    # 3. Génération des embeddings
    print("Génération des embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Shape des embeddings : {embeddings.shape}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # 5. Classification avec MLflow
    mlflow.set_experiment("Eco-Smart_Bonus_CamemBERT")
    
    results = {}
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    best_f1 = 0
    best_clf = None
    best_name = ""
    
    for name, clf in classifiers.items():
        with mlflow.start_run(run_name=f"CamemBERT_{name}"):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            print(f"  {name:25s} — Acc: {acc:.4f} | F1: {f1:.4f}")
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_param("model", name)
            
            results[name] = (acc, f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_clf = clf
                best_name = name
    
    print(f"\n✅ Meilleur classifieur CamemBERT : {best_name} (F1: {best_f1:.4f})")
    
    # Sauvegarde du modèle
    joblib.dump(model, os.path.join(NLP_MODELS_DIR, "camembert_model.pkl"))
    joblib.dump(best_clf, os.path.join(NLP_MODELS_DIR, "camembert_classifier.pkl"))
    print("  ✓ camembert_model.pkl et camembert_classifier.pkl sauvegardés")
    
    return results

# Appeler dans run_nlp_pipeline() à la fin
# if __name__ == "__main__":
#     run_nlp_pipeline()
#     run_camembert_pipeline()  # Bonus
# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
# At the bottom of text_classifier.py — replace the current __main__ block
if __name__ == "__main__":
    run_nlp_pipeline()
    
    # Bonus — only runs if sentence_transformers + torch are available
    try:
        run_camembert_pipeline()
    except (ImportError, OSError) as e:
        print(f"\n⚠️  Bonus CamemBERT ignoré : {e}")
        print("    → Installez PyTorch compatible pour l'activer.")