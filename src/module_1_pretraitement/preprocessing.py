# =============================================================================
# MODULE 1 — EDA, NETTOYAGE ET PRÉTRAITEMENT
# Projet : Eco-Smart Classifier
# Dataset : dataset_ProjetML_2026.csv  (10 500 lignes × 9 colonnes)
#
# AUDIT DU DATASET RÉEL — problèmes identifiés et corrigés :
# ─────────────────────────────────────────────────────────────────────────────
# BUG 1  Conductivite : 198 NaN pour Métal → doivent être imputés par MICE
#        (le code précédent disait "aucun traitement" à tort)
#
# BUG 2  Opacite==55.0 : valeur SENTINELLE (pas un vrai outlier IQR)
#        présente dans toutes catégories → remplacée par NaN avant winsorization
#
# BUG 3  Poids > 1000 : 31 lignes Verre avec Poids jusqu'à 2334 kg
#        → remplacés par NaN en nettoyage physique, avant IQR
#
# BUG 4  fillna(inplace=True) sur slice pandas ≥ 2.0 → ignoré silencieusement
#        → remplacé par affectation directe
#
# BUG 5  Conductivite==0.0 pour Papier/Plastique/Verre = valeur CORRECTE
#        (le code original les convertissait à tort en NaN)
#
# SUPPRIMÉ : PHYSICAL_BOUNDS — dictionnaire défini mais jamais utilisé (dead code)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS & CONSTANTES
# =============================================================================
RAW_DATA_PATH = "data/raw/dataset_ProjetML_2026.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR   = "reports/figures"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR,   exist_ok=True)

NUM_COLS   = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
CAT_COL    = "Source"
TEXT_COL   = "Rapport_Collecte"
TARGET_COL = "Categorie"
CATEGORIES = ["Papier", "Plastique", "Verre", "Métal"]

# Valeur sentinelle identifiée sur Opacite (toutes catégories)
OPACITE_SENTINELLE = 55.0

# Seuil prix aberrant pour des déchets recyclables
PRIX_MAX_VALIDE = 1000.0


# =============================================================================
# 1. CHARGEMENT
# =============================================================================

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Charge le CSV et nettoie les artefacts de lecture.

    Corrections appliquées
    ─────────────────────
    • "nan" string → np.nan  (pandas lit parfois les NaN comme la chaîne "nan"
      quand la colonne est de type object)
    • Aucune conversion de type forcée : les colonnes numériques sont déjà float64
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    df = pd.read_csv(path)

    # Normaliser les colonnes object : chaîne "nan" → vrai NaN
    for col in [TARGET_COL, CAT_COL, TEXT_COL]:
        if col in df.columns:
            df[col] = df[col].replace("nan", np.nan)

    print(f"[load_data] Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  Distribution Categorie : {df[TARGET_COL].value_counts(dropna=False).to_dict()}")
    print(f"  Distribution Source    : {df[CAT_COL].value_counts(dropna=False).to_dict()}")
    return df


# =============================================================================
# 2. EXPLORATION (EDA)
# =============================================================================

def explore_dataset(df: pd.DataFrame) -> None:
    """
    Affiche les statistiques descriptives et sauvegarde 4 figures.

    fig01 — Distribution de la variable cible
    fig02 — Heatmap de corrélation (données numériques brutes)
    fig03 — Boxplots globaux avant nettoyage (montre l'amplitude des outliers)
    fig04 — Boxplots par catégorie avant nettoyage
            (justifie le traitement par groupe)
    """
    print("\n" + "=" * 60)
    print("EXPLORATION DU DATASET")
    print("=" * 60)
    print(f"\nShape : {df.shape}")
    print(f"\nTypes :\n{df.dtypes}")

    print("\nValeurs manquantes (%) :")
    miss = (df.isnull().sum() / len(df) * 100).round(2)
    print(miss[miss > 0].to_string())

    print("\nStatistiques descriptives :")
    print(df[NUM_COLS].describe().round(3).to_string())

    # ── fig01 : Distribution de la cible ─────────────────────────────────
    plt.figure(figsize=(8, 4))
    counts = df[TARGET_COL].value_counts(dropna=False)
    ax = counts.plot(kind="bar", color="steelblue", edgecolor="white")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.title("Distribution de la variable cible")
    plt.xlabel("Catégorie"); plt.ylabel("Effectif")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig01_target_distribution.png"), dpi=120)
    plt.close()
    print("\n  ✓ fig01 sauvegardé")

    # ── fig02 : Heatmap de corrélation ───────────────────────────────────
    numeric_df = df[NUM_COLS].dropna()
    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        plt.title("Matrice de corrélation (avant nettoyage)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig02_correlation.png"), dpi=120)
        plt.close()
        print("  ✓ fig02 sauvegardé")

    # ── fig03 : Boxplots globaux (avant nettoyage) ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), NUM_COLS):
        df[col].dropna().plot(kind="box", ax=ax, title=col,
                              boxprops=dict(color="steelblue"),
                              medianprops=dict(color="red", linewidth=2))
    plt.suptitle("Boxplots globaux — AVANT nettoyage", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig03_boxplots_bruts.png"), dpi=120)
    plt.close()
    print("  ✓ fig03 sauvegardé")

    # ── fig04 : Boxplots par catégorie (avant nettoyage) ─────────────────
    labeled = df.dropna(subset=[TARGET_COL])
    if not labeled.empty:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, col in zip(axes.flatten(), NUM_COLS):
            labeled.boxplot(column=col, by=TARGET_COL, ax=ax)
            ax.set_title(col); ax.set_xlabel("")
        plt.suptitle("Boxplots par catégorie — AVANT nettoyage")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR,
                                 "fig04_boxplots_par_cat_bruts.png"), dpi=120)
        plt.close()
        print("  ✓ fig04 sauvegardé")


# =============================================================================
# 3. ANALYSE DES VALEURS MANQUANTES (MCAR / MAR / MNAR)
# =============================================================================

def classify_missing(df: pd.DataFrame) -> None:
    """
    Identifie le mécanisme de manquance pour chaque colonne avec NaN.

    Méthode
    ────────
    Pour chaque colonne manquante, on compare les moyennes des autres
    variables numériques entre :
    • groupe A : lignes où la valeur EST manquante
    • groupe B : lignes où la valeur est observée

    Si la différence relative est < 5%  → probable MCAR
    Si la différence relative est 5-20% → probable MAR
    Si la différence relative est > 20% → probable MNAR

    Résultats sur ce dataset
    ─────────────────────────
    Poids, Conductivite → MNAR  (diff > 20%)
    Volume, Opacite, Rigidite, Prix_Revente, Source → MAR

    Décision : IterativeImputer (MICE) dans tous les cas
    → C'est la stratégie la plus robuste pour MAR et MNAR.
    """
    print("\n" + "=" * 60)
    print("ANALYSE DES MÉCANISMES DE MANQUANCE")
    print("=" * 60)

    missing_cols = [c for c in df.columns if df[c].isnull().any()]

    for col in missing_cols:
        mask = df[col].isnull()
        pct  = mask.mean() * 100
        print(f"\n  ▸ {col} — {mask.sum()} NaN ({pct:.1f}%)")

        diffs = []
        for num_col in NUM_COLS:
            if num_col == col:
                continue
            g_nan = df.loc[mask,  num_col].mean()
            g_obs = df.loc[~mask, num_col].mean()
            if not (np.isnan(g_nan) or np.isnan(g_obs)):
                diff_rel = abs(g_nan - g_obs) / (abs(g_obs) + 1e-9) * 100
                diffs.append(diff_rel)
                if diff_rel > 10:
                    print(f"      {num_col:15s}: moy(NaN)={g_nan:.2f} "
                          f"vs moy(obs)={g_obs:.2f}  Δ={diff_rel:.1f}% ⚠")

        mean_diff = np.mean(diffs) if diffs else 0
        if mean_diff < 5:
            mec = "MCAR — manquance aléatoire, indépendante des autres variables"
        elif mean_diff < 20:
            mec = "MAR  — manquance liée à d'autres variables observées"
        else:
            mec = "MNAR — manquance informative (liée à la valeur elle-même)"
        print(f"      Mécanisme estimé : {mec}")
        print(f"      Stratégie retenue : IterativeImputer (MICE)")


# =============================================================================
# 4. TRAITEMENT DES OUTLIERS PAR CATÉGORIE
# =============================================================================

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage complet des outliers en 3 étapes ordonnées.

    ══════════════════════════════════════════════════════════════
    ÉTAPE A — Valeurs physiquement impossibles → NaN
    ══════════════════════════════════════════════════════════════
    Ces valeurs sont des erreurs certaines indépendamment
    de la catégorie. On les marque NaN avant tout calcul IQR
    pour ne pas biaiser les bornes de winsorization.

    Poids < 0        : 104 lignes (impossible, un poids est > 0)
    Poids > 1000     : 31 lignes Verre (jusqu'à 2334 kg → erreur)
    Volume < 0       : 245 lignes (impossible)
    Opacite == 55.0  : 93 lignes — VALEUR SENTINELLE identifiée
                       (exactement 55.0 dans toutes les catégories,
                       aucune valeur entre 1.0 et 55.0 → code capteur)
    Opacite > 1      : 4 lignes restantes après sentinelle
                       (opacité physiquement ∈ [0, 1])
    Prix_Revente < 0 : 10 lignes
    Prix_Revente > 1000 : 52 lignes (jusqu'à 9 999€ pour des déchets)

    ══════════════════════════════════════════════════════════════
    ÉTAPE B — Winsorization IQR × 1.5 PAR CATÉGORIE
    ══════════════════════════════════════════════════════════════
    Chaque matériau a ses propres plages physiques. Un poids de
    500 kg est normal pour du Verre mais aberrant pour du Papier.
    → Les bornes IQR sont calculées SÉPARÉMENT par catégorie.
    → Les lignes sans catégorie utilisent l'IQR global.

    Colonnes traitées : Poids, Volume, Opacite, Prix_Revente

    ══════════════════════════════════════════════════════════════
    ÉTAPE C — Colonnes sans winsorization
    ══════════════════════════════════════════════════════════════
    Conductivite :
      • Papier / Plastique / Verre : exactement 0.0 → CORRECT
        (matériaux non conducteurs, pas une erreur)
      • Métal : [0.80 – 1.0], 198 NaN → imputés par MICE à l'étape 5
        ATTENTION : ne pas mettre 0.0 en NaN pour Métal (valeur hors plage)
    Rigidite :
      • Valeurs entières dans [1, 2, 3, 4, 5, 7, 8, 9, 10]
      • Aucun outlier, plage propre → aucun traitement
    """
    print("\n" + "=" * 60)
    print("TRAITEMENT DES OUTLIERS (PAR CATÉGORIE)")
    print("=" * 60)

    df_clean = df.copy()

    # ─────────────────────────────────────────────────────────────────────
    # Fonction interne : Winsorization IQR par catégorie
    # ─────────────────────────────────────────────────────────────────────
    def winsorize_by_category(df_in: pd.DataFrame, col: str,
                              factor: float = 1.5) -> tuple:
        """
        Clipping IQR × factor par groupe catégorie, puis IQR global
        pour les lignes sans catégorie connue.

        Retourne (df_modifié, n_total_valeurs_clippées).
        """
        total_clipped = 0

        # ── Par catégorie connue ──────────────────────────────────────
        for cat in sorted(df_in[TARGET_COL].dropna().unique()):
            mask_cat = df_in[TARGET_COL] == cat
            subset   = df_in.loc[mask_cat, col].dropna()
            if len(subset) < 4:
                continue

            Q1  = subset.quantile(0.25)
            Q3  = subset.quantile(0.75)
            IQR = Q3 - Q1
            lo  = Q1 - factor * IQR
            hi  = Q3 + factor * IQR

            avant  = df_in.loc[mask_cat, col].copy()
            df_in.loc[mask_cat, col] = df_in.loc[mask_cat, col].clip(lo, hi)
            n_clip = int((avant != df_in.loc[mask_cat, col]).sum())
            total_clipped += n_clip
            print(f"    {cat:12s}: Q1={Q1:7.2f}  Q3={Q3:7.2f}  "
                  f"IQR={IQR:7.2f}  bornes [{lo:.2f}, {hi:.2f}]  "
                  f"→ {n_clip} clippés")

        # ── Lignes sans catégorie → IQR global ───────────────────────
        mask_no_cat = df_in[TARGET_COL].isna()
        if mask_no_cat.any():
            ref    = df_in.loc[~mask_no_cat, col].dropna()
            Q1g    = ref.quantile(0.25)
            Q3g    = ref.quantile(0.75)
            IQRg   = Q3g - Q1g
            lo_g   = Q1g - factor * IQRg
            hi_g   = Q3g + factor * IQRg
            avant  = df_in.loc[mask_no_cat, col].copy()
            df_in.loc[mask_no_cat, col] = df_in.loc[mask_no_cat, col].clip(lo_g, hi_g)
            n_clip_g = int((avant != df_in.loc[mask_no_cat, col]).sum())
            total_clipped += n_clip_g
            print(f"    {'(sans cat.)':12s}: IQR global  "
                  f"bornes [{lo_g:.2f}, {hi_g:.2f}]  → {n_clip_g} clippés")

        return df_in, total_clipped

    # ═════════════════════════════════════════════════════════════════════
    # ÉTAPE A : Nettoyage physique
    # ═════════════════════════════════════════════════════════════════════
    print("\n  ── ÉTAPE A : Valeurs physiquement impossibles → NaN ──")

    # Poids négatif
    n = int((df_clean["Poids"] < 0).sum())
    df_clean.loc[df_clean["Poids"] < 0, "Poids"] = np.nan
    print(f"\n  [Poids]  {n} valeurs < 0 remplacées par NaN")

    # Poids > 1000 (uniquement Verre, jusqu'à 2334 kg)
    n2 = int((df_clean["Poids"] > 1000).sum())
    df_clean.loc[df_clean["Poids"] > 1000, "Poids"] = np.nan
    print(f"  [Poids]  {n2} valeurs > 1000 kg remplacées par NaN "
          f"(Verre : erreurs de pesée)")

    # Volume négatif
    n = int((df_clean["Volume"] < 0).sum())
    df_clean.loc[df_clean["Volume"] < 0, "Volume"] = np.nan
    print(f"\n  [Volume] {n} valeurs < 0 remplacées par NaN")

    # Opacite == 55.0 (valeur sentinelle capteur, exacte dans toutes catégories)
    n_sent = int((df_clean["Opacite"] == OPACITE_SENTINELLE).sum())
    df_clean.loc[df_clean["Opacite"] == OPACITE_SENTINELLE, "Opacite"] = np.nan
    print(f"\n  [Opacite] {n_sent} valeurs sentinelles (==55.0) → NaN")

    # Opacite > 1 restantes (erreurs de mesure, opacité ∈ [0,1])
    n_sup1 = int((df_clean["Opacite"] > 1).sum())
    df_clean.loc[df_clean["Opacite"] > 1, "Opacite"] = np.nan
    print(f"  [Opacite] {n_sup1} valeurs > 1 restantes → NaN "
          f"(opacité physiquement ∈ [0,1])")

    # Prix négatif
    n = int((df_clean["Prix_Revente"] < 0).sum())
    df_clean.loc[df_clean["Prix_Revente"] < 0, "Prix_Revente"] = np.nan
    print(f"\n  [Prix_Revente] {n} valeurs < 0 remplacées par NaN")

    # Prix > 1000 (jusqu'à 9 999 € — aberrant pour des déchets)
    n = int((df_clean["Prix_Revente"] > PRIX_MAX_VALIDE).sum())
    df_clean.loc[df_clean["Prix_Revente"] > PRIX_MAX_VALIDE, "Prix_Revente"] = np.nan
    print(f"  [Prix_Revente] {n} valeurs > {PRIX_MAX_VALIDE:.0f} € remplacées par NaN")

    # ═════════════════════════════════════════════════════════════════════
    # ÉTAPE B : Winsorization IQR × 1.5 par catégorie
    # ═════════════════════════════════════════════════════════════════════
    print("\n  ── ÉTAPE B : Winsorization IQR × 1.5 par catégorie ──")

    print("\n  [Poids]")
    df_clean, n = winsorize_by_category(df_clean, "Poids")
    print(f"    Total clippés : {n}")

    print("\n  [Volume]")
    df_clean, n = winsorize_by_category(df_clean, "Volume")
    print(f"    Total clippés : {n}")

    print("\n  [Opacite]")
    df_clean, n = winsorize_by_category(df_clean, "Opacite")
    print(f"    Total clippés : {n}")

    print("\n  [Prix_Revente]")
    df_clean, n = winsorize_by_category(df_clean, "Prix_Revente")
    print(f"    Total clippés : {n}")

    # ═════════════════════════════════════════════════════════════════════
    # ÉTAPE C : Colonnes sans traitement
    # ═════════════════════════════════════════════════════════════════════
    print("\n  ── ÉTAPE C : Colonnes sans traitement ──")
    print("  [Conductivite]")
    print("    Papier/Plastique/Verre : 0.0 → valeur CORRECTE (non conducteurs)")
    print("    Métal : [0.80–1.0], NaN présents → seront imputés par MICE")
    cond_nan_metal = int(
        df_clean[df_clean[TARGET_COL] == "Métal"]["Conductivite"].isna().sum()
    )
    print(f"    NaN Conductivite pour Métal : {cond_nan_metal}")

    print("  [Rigidite] → valeurs entières [1–10] propres, 0 outlier")

    # ═════════════════════════════════════════════════════════════════════
    # BILAN
    # ═════════════════════════════════════════════════════════════════════
    print("\n  ── Bilan NaN après traitement des outliers ──")
    for col in NUM_COLS:
        n_nan = int(df_clean[col].isna().sum())
        pct   = n_nan / len(df_clean) * 100
        flag  = " ← sera imputé par MICE" if n_nan > 0 else ""
        print(f"    {col:15s}: {n_nan:5d} NaN ({pct:.1f}%){flag}")

    return df_clean


# =============================================================================
# 5. COMPARAISON DES STRATÉGIES D'IMPUTATION
# =============================================================================

def compare_imputation_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare 3 stratégies d'imputation et retourne le résultat MICE.

    Stratégie 1 — Médiane
    ─────────────────────
    Chaque NaN est remplacé par la médiane de sa colonne.
    + Rapide, robuste aux outliers résiduels
    − Ignore totalement les corrélations entre colonnes
    − Introduit le même biais pour tous les NaN d'une colonne

    Stratégie 2 — KNN Imputer (k=5)
    ─────────────────────────────────
    Chaque NaN est remplacé par la moyenne des k voisins les plus proches
    (similarité calculée sur les colonnes observées).
    + Exploite la structure locale des données
    − Coûteux en mémoire (O(n²) distances) sur 10 000+ lignes
    − Sensible au choix de k

    Stratégie 3 — MICE (IterativeImputer)
    ───────────────────────────────────────
    Chaque colonne avec NaN est modélisée par une régression sur les autres.
    Le processus itère jusqu'à convergence.
    + Capture les corrélations entre toutes les variables
    + Adapté aux mécanismes MAR et MNAR identifiés
    + Résultat cohérent avec la structure jointe des données
    − Plus lent (10 itérations ici)
    − Moins stable si très peu de données disponibles

    Pourquoi MICE est retenu
    ─────────────────────────
    Les variables physiques sont corrélées par catégorie :
    ex. Poids ↑ ↔ Rigidite ↑ ↔ Prix_Revente ↑ pour le Métal.
    MICE capture ces relations et produit des imputations réalistes.
    """
    print("\n" + "=" * 60)
    print("COMPARAISON DES STRATÉGIES D'IMPUTATION")
    print("=" * 60)

    df_num = df[NUM_COLS].copy()
    nan_total = int(df_num.isnull().sum().sum())
    print(f"  NaN à imputer (toutes colonnes) : {nan_total}")
    for col in NUM_COLS:
        n = int(df_num[col].isnull().sum())
        if n > 0:
            print(f"    {col:15s}: {n} NaN")

    # ── Stratégie 1 : Médiane ─────────────────────────────────────────
    # CORRECTION BUG 4 : ne pas utiliser fillna(inplace=True) sur une slice
    # pandas ≥ 2.0 ignore silencieusement cette opération
    df_median = df_num.copy()
    for col in NUM_COLS:
        median_val = df_median[col].median()
        df_median[col] = df_median[col].fillna(median_val)   # ← affectation directe
    print(f"\n  [1] Médiane       — NaN restants : {df_median.isnull().sum().sum()}")
    print(f"      + Rapide, robuste aux outliers résiduels")
    print(f"      − Ignore les corrélations inter-colonnes")

    # ── Stratégie 2 : KNN Imputer ─────────────────────────────────────
    knn    = KNNImputer(n_neighbors=5)
    df_knn = pd.DataFrame(knn.fit_transform(df_num), columns=NUM_COLS)
    print(f"\n  [2] KNN (k=5)     — NaN restants : {df_knn.isnull().sum().sum()}")
    print(f"      + Exploite la similarité locale entre lignes")
    print(f"      − Coûteux sur grand dataset (O(n²) distances)")

    # ── Stratégie 3 : MICE ────────────────────────────────────────────
    mice    = IterativeImputer(random_state=42, max_iter=10, tol=1e-3)
    df_iter = pd.DataFrame(mice.fit_transform(df_num), columns=NUM_COLS)
    print(f"\n  [3] MICE          — NaN restants : {df_iter.isnull().sum().sum()}")
    print(f"      + Modélise chaque variable par régression sur les autres")
    print(f"      + Respecte les corrélations MAR/MNAR détectées")

    print(f"\n  → Stratégie retenue : MICE (IterativeImputer) ✓")

    # Vérification post-imputation : les moyennes par catégorie sont-elles cohérentes ?
    df_check = df[[TARGET_COL]].copy()
    for col in NUM_COLS:
        df_check[col] = df_iter[col].values
    print("\n  Vérification cohérence post-imputation (moyennes par catégorie) :")
    labeled = df_check.dropna(subset=[TARGET_COL])
    print(labeled.groupby(TARGET_COL)[NUM_COLS].mean().round(3).to_string())

    return df_iter


# =============================================================================
# 6. VISUALISATION POST-NETTOYAGE
# =============================================================================

def plot_after_cleaning(df_clean: pd.DataFrame) -> None:
    """
    Sauvegarde les boxplots APRÈS nettoyage (fig05).
    Permet de comparer visuellement avec fig04 (avant nettoyage).
    """
    labeled = df_clean.dropna(subset=[TARGET_COL])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.flatten(), NUM_COLS):
        labeled.boxplot(column=col, by=TARGET_COL, ax=ax)
        ax.set_title(col); ax.set_xlabel("")
    plt.suptitle("Boxplots par catégorie — APRÈS nettoyage")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,
                             "fig05_boxplots_par_cat_propres.png"), dpi=120)
    plt.close()
    print("  ✓ fig05 sauvegardé")


# =============================================================================
# 7. ENCODAGE ET NORMALISATION
# =============================================================================

def encode_and_scale(df: pd.DataFrame):
    """
    Encode les variables catégorielles et normalise les numériques.

    1. Label Encoding de la cible
    ────────────────────────────
    Categorie → entier : Métal=0, Papier=1, Plastique=2, Verre=3
    Les lignes avec Categorie NaN reçoivent Categorie_enc = NaN
    (exclues du split supervisé, utilisables en semi-supervisé)

    2. One-Hot Encoding de Source
    ──────────────────────────────
    Les NaN de Source sont remplacés par "Inconnu" avant l'OHE.
    handle_unknown="ignore" → les nouvelles modalités en prod
    donnent un vecteur de zéros (pas d'erreur).
    Résultat : 5 colonnes binaires Source_*

    3. StandardScaler sur les 6 colonnes numériques
    ─────────────────────────────────────────────────
    Chaque colonne est centrée (mean=0) et réduite (std=1).
    Nécessaire car les amplitudes sont très différentes :
    Poids [0–300], Conductivite [0–1], Prix_Revente [0–30]
    → Sans scaling, un KNN ou SVM serait dominé par Poids.

    Objets sauvegardés via joblib
    ──────────────────────────────
    scaler.pkl / onehot_encoder.pkl / label_encoder.pkl
    → Indispensables pour transformer les nouvelles données
      avec les MÊMES paramètres (mean, std, catégories OHE)
    """
    print("\n" + "=" * 60)
    print("ENCODAGE ET NORMALISATION")
    print("=" * 60)

    df_enc = df.copy()

    # ── 1. Label Encoding de la cible ─────────────────────────────────
    le = LabelEncoder()
    mask_labeled = df_enc[TARGET_COL].notna()
    df_enc["Categorie_enc"] = np.nan
    df_enc.loc[mask_labeled, "Categorie_enc"] = le.fit_transform(
        df_enc.loc[mask_labeled, TARGET_COL]
    )
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  LabelEncoder   : {mapping}")

    # ── 2. One-Hot Encoding de Source ─────────────────────────────────
    df_enc[CAT_COL] = df_enc[CAT_COL].fillna("Inconnu")
    ohe      = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_arr  = ohe.fit_transform(df_enc[[CAT_COL]])
    ohe_cols = [f"Source_{c}" for c in ohe.categories_[0]]
    df_ohe   = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_enc.index)
    df_enc   = pd.concat([df_enc, df_ohe], axis=1)
    print(f"  OneHotEncoder  : {ohe_cols}")

    # ── 3. StandardScaler ─────────────────────────────────────────────
    scaler = StandardScaler()
    df_enc[NUM_COLS] = scaler.fit_transform(df_enc[NUM_COLS])
    print(f"  StandardScaler appliqué sur : {NUM_COLS}")
    for col, m, s in zip(NUM_COLS, scaler.mean_.round(3), scaler.scale_.round(3)):
        print(f"    {col:15s}: mean={m:8.3f}  std={s:.3f}")

    # ── Sauvegarde ────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
    joblib.dump(ohe,    os.path.join(PROCESSED_DIR, "onehot_encoder.pkl"))
    joblib.dump(le,     os.path.join(PROCESSED_DIR, "label_encoder.pkl"))
    print("  ✓ scaler.pkl, onehot_encoder.pkl, label_encoder.pkl sauvegardés")

    return df_enc, scaler, ohe, le


# =============================================================================
# 8. SPLIT TRAIN / VAL / TEST (70 / 15 / 15)
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Split stratifié 70 / 15 / 15 sur les lignes labellisées uniquement.

    Stratifié → chaque split conserve la même proportion de chaque
    catégorie que l'ensemble complet (pas de biais de sélection).

    Les 514 lignes sans catégorie sont EXCLUES ici mais sauvegardées
    dans clean_data.csv pour un usage semi-supervisé éventuel.

    Double split
    ─────────────
    1er split : 85% train_val  |  15% test
    2ème split : sur train_val → 82.35% train  |  17.65% val
                 17.65% × 85% ≈ 15% du total → ratio 70/15/15 ✓
    """
    print("\n" + "=" * 60)
    print("SPLIT TRAIN / VAL / TEST (70 / 15 / 15)")
    print("=" * 60)

    df_labeled = df.dropna(subset=["Categorie_enc"]).copy()

    # Exclure les colonnes non numériques du split
    cols_drop = [TARGET_COL, "Categorie_enc", TEXT_COL]
    X = df_labeled.drop(columns=[c for c in cols_drop if c in df_labeled.columns])
    y = df_labeled["Categorie_enc"].astype(int).values

    print(f"  Lignes labellisées : {len(df_labeled)}")
    print(f"  Features (X)       : {X.shape[1]} colonnes")

    # ── Split 1 : 85% train_val | 15% test ────────────────────────────
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, test_idx = next(sss1.split(X, y))
    X_tv,    X_test  = X.iloc[tv_idx],    X.iloc[test_idx]
    y_tv,    y_test  = y[tv_idx],         y[test_idx]

    # ── Split 2 : 70% train | 15% val ─────────────────────────────────
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    tr_idx, val_idx  = next(sss2.split(X_tv, y_tv))
    X_train, X_val   = X_tv.iloc[tr_idx], X_tv.iloc[val_idx]
    y_train, y_val   = y_tv[tr_idx],      y_tv[val_idx]

    # ── Tableau de vérification de la stratification ───────────────────
    from sklearn.preprocessing import LabelEncoder as _LE
    _le = _LE(); _le.fit(y)   # classes déjà encodées
    classes = ["Métal(0)", "Papier(1)", "Plastique(2)", "Verre(3)"]
    header  = f"  {'Split':<8} {'Total':>6}  " + "  ".join(f"{c:>12}" for c in classes)
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for name, y_s in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = [int((y_s == i).sum()) for i in range(4)]
        pcts   = [f"{c}({c/len(y_s)*100:.0f}%)" for c in counts]
        print(f"  {name:<8} {len(y_s):>6}  " + "  ".join(f"{p:>12}" for p in pcts))

    # ── Sauvegarde ────────────────────────────────────────────────────
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_val.to_csv(  os.path.join(PROCESSED_DIR, "X_val.csv"),   index=False)
    X_test.to_csv( os.path.join(PROCESSED_DIR, "X_test.csv"),  index=False)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    print("\n  ✓ 6 fichiers sauvegardés dans data/processed/")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 9. PIPELINE PRINCIPAL
# =============================================================================

def run_preprocessing(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Exécute le pipeline complet dans l'ordre correct.

    ORDRE CRITIQUE
    ─────────────────────────────────────────────────────────────
    outliers AVANT imputation
    → Si on imputait en premier, les outliers (Poids=-99, Prix=9999)
      fausseraient les statistiques utilisées par MICE.

    clean_data.csv sauvegardé AVANT scaling
    → Le multimodal pipeline appliquera scaler.transform() lui-même.
      Sauvegarder après scaling causerait une double normalisation.
    ─────────────────────────────────────────────────────────────
    """
    print("\n" + "=" * 60)
    print("PIPELINE COMPLET — Eco-Smart Classifier (Module 1)")
    print("=" * 60)

    # 1. Chargement
    df = load_data(path=raw_path)

    # 2. EDA
    explore_dataset(df)

    # 3. Analyse des manquances
    classify_missing(df)

    # 4. Outliers (AVANT imputation)
    df = handle_outliers(df)

    # 5. Visualisation après nettoyage
    plot_after_cleaning(df)

    # 6. Imputation MICE
    df_imputed = compare_imputation_strategies(df)
    for col in NUM_COLS:
        df[col] = df_imputed[col].values

    # 7. Sauvegarde BRUTE (avant scaling) — pour multimodal pipeline
    le_temp = LabelEncoder()
    mask    = df[TARGET_COL].notna()
    df["Categorie_enc"] = np.nan
    df.loc[mask, "Categorie_enc"] = le_temp.fit_transform(df.loc[mask, TARGET_COL])
    clean_path = os.path.join(PROCESSED_DIR, "clean_data.csv")
    df.to_csv(clean_path, index=False)
    print(f"\n  ✅ clean_data.csv sauvegardé (brut, avant scaling)")

    # 8. Encodage + Scaling
    df_final, scaler, ohe, le = encode_and_scale(df)

    # 9. Split
    split_data(df_final)

    # 10. Résumé
    print(f"\n{'=' * 60}")
    print("RÉSUMÉ PIPELINE")
    print(f"{'=' * 60}")
    print(f"  Shape finale           : {df_final.shape}")
    print(f"  Lignes labellisées     : {int(df[TARGET_COL].notna().sum())}")
    print(f"  Lignes non labellisées : {int(df[TARGET_COL].isna().sum())}")
    print(f"  Features finales       : {df_final.shape[1]} colonnes")
    print("\n  Moyennes par catégorie (valeurs brutes, post-nettoyage) :")
    check = df.dropna(subset=[TARGET_COL])
    print(check.groupby(TARGET_COL)[NUM_COLS].mean().round(3).to_string())
    print("\n  Pipeline terminé avec succès ✓")

    return df_final


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    run_preprocessing()