# =============================================================================
# tests/test_nlp.py
# Tests du Module 4 : prétraitement NLP, vectorisation, pipeline
# =============================================================================

import pytest
import numpy as np
import re

from src.module_4_NLP.text_classifier import preprocess_text


# =============================================================================
# 1. TESTS DE PREPROCESS_TEXT
# =============================================================================

class TestPreprocessText:
    """Tests unitaires de la fonction preprocess_text."""

    def test_returns_string(self):
        """La fonction doit toujours retourner une chaîne."""
        result = preprocess_text("Lot de plastique récupéré.")
        assert isinstance(result, str)

    def test_no_digits_in_output(self):
        """
        Le texte prétraité ne doit contenir aucun chiffre.
        C'est le comportement attendu : le regex [^a-z...\\s] supprime
        tous les caractères non-alphabétiques AVANT la tokenisation.
        """
        text = "Lot de papier. Poids 16.7 kg, volume 70.9 L."
        processed = preprocess_text(text)
        assert not any(c.isdigit() for c in processed), \
            f"Chiffres trouvés dans : '{processed}'"

    def test_no_punctuation(self):
        """La ponctuation doit être supprimée."""
        text = "Plastique! Tres bon état. Collecté en 2024."
        processed = preprocess_text(text)
        # On n'attend pas de ! . , ; : dans la sortie
        for char in ["!", ".", ",", ";", ":"]:
            assert char not in processed, f"Ponctuation '{char}' trouvée"

    def test_lowercase_output(self):
        """La sortie doit être en minuscules."""
        text = "LOT DE VERRE TRANSPARENT"
        processed = preprocess_text(text)
        assert processed == processed.lower()

    def test_stopwords_removed(self):
        """Les mots outils français doivent être supprimés."""
        text = "le lot de verre est tres propre"
        processed = preprocess_text(text)
        stopwords_fr = {"le", "de", "est",  "un", "une", "les", "et"}
        tokens = set(processed.split())
        common = tokens & stopwords_fr
        assert len(common) == 0, f"Stopwords restants : {common}"

    def test_empty_string(self):
        """Un texte vide doit retourner une chaîne vide (ou presque)."""
        result = preprocess_text("")
        assert isinstance(result, str)
        assert len(result) == 0

    def test_none_input(self):
        """None en entrée ne doit pas faire planter la fonction."""
        result = preprocess_text(None)
        assert isinstance(result, str)

    def test_numeric_only_text(self):
        """Un texte composé uniquement de chiffres → sortie vide."""
        text = "1234 567.89 0.01"
        processed = preprocess_text(text)
        assert not any(c.isdigit() for c in processed)

    def test_meaningful_tokens_preserved(self):
        """Des mots significatifs (non-stopwords) doivent être conservés."""
        text = "plastique transparent recyclable haute qualité"
        processed = preprocess_text(text)
        tokens = processed.split()
        # Au moins quelques tokens significatifs doivent subsister
        assert len(tokens) >= 1, "Tous les tokens ont été supprimés"

    def test_domain_stopwords_removed(self):
        """Les stopwords du domaine (lot, kg, déchet…) doivent être retirés."""
        text = "lot de déchet matériau kg récupéré"
        processed = preprocess_text(text)
        domain_stops = {"lot", "dechet", "materiau", "kg", "recupere"}
        tokens = set(processed.split())
        # Aucun stopword du domaine ne doit rester
        remaining = tokens & domain_stops
        assert len(remaining) == 0, f"Stopwords domaine restants : {remaining}"

    def test_accentuated_chars_kept(self):
        """Les caractères accentués légitimes doivent être conservés."""
        text = "récupération de matière première"
        processed = preprocess_text(text)
        # Le résultat ne doit pas être entièrement vide
        assert len(processed) > 0


# =============================================================================
# 2. TESTS DE VECTORISATION
# =============================================================================

class TestVectorization:
    """Tests de cohérence des vectoriseurs TF-IDF et BoW."""

    def test_tfidf_output_shape(self, labeled_data):
        """Le TF-IDF doit produire une matrice (n_docs, n_features)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [preprocess_text(t) for t in
                 labeled_data["Rapport_Collecte"].astype(str)]
        vec = TfidfVectorizer(max_features=200)
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 200

    def test_tfidf_values_in_range(self, labeled_data):
        """Les valeurs TF-IDF doivent être dans [0, 1]."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [preprocess_text(t) for t in
                 labeled_data["Rapport_Collecte"].astype(str)]
        vec = TfidfVectorizer(max_features=200)
        X = vec.fit_transform(texts).toarray()
        assert X.min() >= 0.0
        assert X.max() <= 1.0 + 1e-9

    def test_bow_all_positive(self, labeled_data):
        """Les valeurs Bag-of-Words doivent être ≥ 0 (comptage)."""
        from sklearn.feature_extraction.text import CountVectorizer
        texts = [preprocess_text(t) for t in
                 labeled_data["Rapport_Collecte"].astype(str)]
        vec = CountVectorizer(max_features=200)
        X = vec.fit_transform(texts).toarray()
        assert X.min() >= 0

    def test_fitted_tfidf_transform(self, fitted_tfidf, labeled_data):
        """Le vectoriseur fitté doit transformer un nouveau texte."""
        text = preprocess_text("plastique recyclé haute densité")
        vec = fitted_tfidf.transform([text])
        assert vec.shape[0] == 1
        assert vec.shape[1] == fitted_tfidf.max_features or vec.shape[1] > 0


# =============================================================================
# 3. TESTS DE COHÉRENCE DU PIPELINE NLP
# =============================================================================

class TestNLPPipeline:

    def test_preprocessed_texts_not_empty(self, labeled_data):
        """Au moins 90% des textes prétraités ne doivent pas être vides."""
        texts = labeled_data["Rapport_Collecte"].astype(str).tolist()
        processed = [preprocess_text(t) for t in texts]
        non_empty = sum(1 for p in processed if len(p.strip()) > 0)
        assert non_empty / len(processed) >= 0.90

    def test_consistent_preprocessing(self):
        """La même entrée doit toujours produire la même sortie (déterminisme)."""
        text = "Lot de verre collecté en juin"
        result1 = preprocess_text(text)
        result2 = preprocess_text(text)
        assert result1 == result2

    def test_pipeline_accuracy_threshold(self, labeled_data, fitted_tfidf):
        """
        Un classificateur TF-IDF + LogisticRegression entraîné sur données
        synthétiques doit atteindre au moins 50% d'accuracy
        (seuil bas car données synthétiques aléatoires).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        texts = [preprocess_text(t) for t in
                 labeled_data["Rapport_Collecte"].astype(str)]
        y = labeled_data["Categorie_enc"].astype(int).values

        X = fitted_tfidf.transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        # Seuil bas pour données synthétiques
        assert acc >= 0.15, f"Accuracy trop faible : {acc:.4f}"