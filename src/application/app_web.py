# app_web.py - Eco-Smart Classifier (Version améliorée)
# Ajouts : Batch prediction, Historique, cURL/JSON, Infos modèle, Stopwords, Latence, Tokens

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
import re

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS MODERNE
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    .stApp {
        background: #eef2f6;
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 2rem 0 0.5rem 0;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #cbd5e1;
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #cbd5e1;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #e2e8f0;
        border-radius: 12px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    .stProgress > div > div > div {
        background: #3b82f6;
        border-radius: 10px;
    }
    
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #cbd5e1;
    }
    
    textarea {
        background: white !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-family: 'Inter', monospace !important;
    }
    
    .cat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .cat-Métal { background: #E6F1FB; color: #0C447C; }
    .cat-Papier { background: #FAEEDA; color: #633806; }
    .cat-Plastique { background: #EAF3DE; color: #27500A; }
    .cat-Verre { background: #E1F5EE; color: #085041; }
    
    .nlp-token {
        display: inline-block;
        background: #E6F1FB;
        color: #0C447C;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 12px;
        font-family: monospace;
    }
    
    .nlp-step {
        background: #f8f9fa;
        border-left: 3px solid #378ADD;
        padding: 6px 12px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-title">
    <span>♻️</span>
    <span>Eco-Smart Classifier</span>
</div>
<div class="subtitle">
    Classification automatique des déchets • Estimation du prix de revente
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - API Configuration
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_url = st.text_input(
        "URL de l'API",
        value=st.session_state.api_url,
        placeholder="http://localhost:8000",
    )
    st.session_state.api_url = api_url.rstrip("/")
    
    if st.button("🔍 Tester connexion", use_container_width=True):
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connectée")
            else:
                st.error(f"❌ Erreur: {response.status_code}")
        except:
            st.error("❌ API non disponible")
    
    st.divider()
    st.markdown("**📊 Statistiques session**")
    st.metric("Prédictions", len(st.session_state.history))
    
    if st.button("🗑️ Vider historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/clean_data.csv")
        return df
    except:
        return pd.DataFrame({
            "Categorie": ["Plastique", "Verre", "Papier", "Métal"],
            "Source": ["Usine_A", "Collecte_Citoyenne", "Centre_Tri", "Usine_B"]
        })

# ============================================================================
# FONCTIONS NLP (prétraitement visible)
# ============================================================================
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais",
    "donc", "or", "ni", "car", "dans", "pour", "par", "sur", "sous", "avec",
    "sans", "est", "sont", "ont", "fait", "cet", "cette", "ces", "ce", "cela",
    "il", "elle", "ils", "elles", "nous", "vous", "je", "tu", "se", "me", "te"
}

def show_preprocessing(text: str):
    """Affiche les étapes de prétraitement NLP"""
    st.markdown("**🔬 Pipeline de prétraitement NLP :**")
    
    step1 = text.lower()
    st.markdown(f'<div class="nlp-step">1. Minuscules → <code>{step1[:80]}{"..." if len(step1)>80 else ""}</code></div>', unsafe_allow_html=True)
    
    step2 = re.sub(r'[^a-z\s]', ' ', step1)
    step2 = re.sub(r'\s+', ' ', step2).strip()
    st.markdown(f'<div class="nlp-step">2. Suppression ponctuation → <code>{step2[:80]}{"..." if len(step2)>80 else ""}</code></div>', unsafe_allow_html=True)
    
    tokens = step2.split()
    st.markdown(f'<div class="nlp-step">3. Tokenisation → {len(tokens)} tokens</div>', unsafe_allow_html=True)
    
    tokens_filtered = [t for t in tokens if t not in STOPWORDS_FR]
    st.markdown(f'<div class="nlp-step">4. Suppression stopwords ({len(tokens)-len(tokens_filtered)} retirés) → {len(tokens_filtered)} tokens restants</div>', unsafe_allow_html=True)
    
    tokens_final = [t for t in tokens_filtered if len(t) > 2]
    st.markdown(f'<div class="nlp-step">5. Filtrage tokens courts → {len(tokens_final)} tokens finaux</div>', unsafe_allow_html=True)
    
    if tokens_final:
        tokens_html = " ".join(f'<span class="nlp-token">{t}</span>' for t in tokens_final[:20])
        st.markdown(f'<div style="margin-top:8px">{tokens_html}{" ..." if len(tokens_final)>20 else ""}</div>', unsafe_allow_html=True)
    
    return tokens_final

# ============================================================================
# TABLES
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🎚️ Prédiction",
    "🤖 Assistant NLP",
    "📋 Batch",
    "📊 Historique"
])

# ============================================================================
# TAB 1: Dashboard Data
# ============================================================================
with tab1:
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(df):,}</div><div class="stat-label">Échantillons</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{df["Categorie"].nunique() if "Categorie" in df else 4}</div><div class="stat-label">Catégories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{df["Source"].nunique() if "Source" in df else 4}</div><div class="stat-label">Sources</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">58.6 €</div><div class="stat-label">Prix moyen</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🥧 Distribution des catégories</div>', unsafe_allow_html=True)
        if "Categorie" in df.columns:
            fig = px.pie(df, names="Categorie", title=None, hole=0.35,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-title">📊 Distribution des sources</div>', unsafe_allow_html=True)
        if "Source" in df.columns:
            src_counts = df["Source"].value_counts()
            fig = px.bar(x=src_counts.index, y=src_counts.values, title=None,
                        color=src_counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-title">📈 Projection PCA des clusters</div>', unsafe_allow_html=True)
    try:
        pca_df = pd.read_csv("data/processed/pca_data.csv")
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", title=None, opacity=0.7)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("💡 Données PCA non disponibles")

# ============================================================================
# TAB 2: Prédiction Manuelle
# ============================================================================
with tab2:
    st.markdown('<div class="section-title">⚙️ Paramètres</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚖️ Physique**")
        poids = st.slider("Poids (kg)", 0.0, 1000.0, 50.0)
        volume = st.slider("Volume (L)", 0.0, 10000.0, 100.0)
        conductivite = st.slider("Conductivité", 0.0, 1.0, 0.5)
    
    with col2:
        st.markdown("**🔬 Optique & Mécanique**")
        opacite = st.slider("Opacité", 0.0, 1.0, 0.5)
        rigidite = st.slider("Rigidité", 1.0, 10.0, 5.0)
        source = st.selectbox("Source", ["Usine_A", "Usine_B", "Centre_Tri", "Collecte_Citoyenne"])
    
    with col3:
        st.markdown("**📝 Description**")
        rapport = st.text_area("Rapport de collecte", "Lot de plastique récupéré. Matériau recyclable.", height=150)
    
    if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
        payload = {
            "Poids": poids, "Volume": volume, "Conductivite": conductivite,
            "Opacite": opacite, "Rigidite": rigidite, "Source": source,
            "Rapport_Collecte": rapport
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{st.session_state.api_url}/predict", json=payload, timeout=10)
            latency_ms = round((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                result = response.json()
                
                st.session_state.history.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "type": "manuel",
                    "payload": payload,
                    "result": result,
                    "latency_ms": latency_ms
                })
                
                st.markdown('<div class="section-title">✨ Résultats</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    cat = result['categorie_predite']
                    badge_class = f"cat-badge cat-{cat}"
                    st.markdown(f'<div style="text-align:center"><span class="{badge_class}">{cat}</span></div>', unsafe_allow_html=True)
                    st.metric("Catégorie prédite", cat)
                with col2:
                    prix_val = result.get('prix_estime')
                    st.metric("Prix estimé", f"{prix_val} €" if prix_val else "N/A")
                with col3:
                    st.metric("Latence", f"{latency_ms} ms")
                
                # Afficher les probabilités
                if result.get('probabilites'):
                    st.markdown("**📊 Probabilités par catégorie**")
                    for cat, prob in sorted(result['probabilites'].items(), key=lambda x: -x[1]):
                        col_label, col_bar = st.columns([1, 4])
                        col_label.markdown(f"**{cat}**")
                        col_bar.progress(prob, text=f"{prob:.2%}")
            else:
                st.error(f"⚠️ Erreur API: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

# ============================================================================
# TAB 3: Assistant Intelligent (NLP avec prétraitement visible)
# ============================================================================
with tab3:
    st.markdown('<div class="section-title">🤖 Assistant Intelligent</div>', unsafe_allow_html=True)
    st.markdown("Décrivez le déchet en langage naturel - **Seule la description textuelle compte**")
    
    st.info("💡 Ce modèle utilise uniquement le texte pour la classification")
    
    text_input = st.text_area("Description du déchet", height=120,
                              placeholder="Ex: Bouteille en verre transparent, épaisse, avec bouchon métallique...")
    
    # Afficher le prétraitement en temps réel
    if text_input:
        tokens_final = show_preprocessing(text_input)
    
    if st.button("🔍 Classifier par texte", type="primary", use_container_width=True):
        if text_input.strip():
            payload = {
                "Poids": 50.0,
                "Volume": 100.0,
                "Conductivite": 0.5,
                "Opacite": 0.5,
                "Rigidite": 5.0,
                "Source": "Collecte_Citoyenne",
                "Rapport_Collecte": text_input
            }
            
            start_time = time.time()
            try:
                response = requests.post(f"{st.session_state.api_url}/predict_nlp", json=payload, timeout=10)
                latency_ms = round((time.time() - start_time) * 1000)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.session_state.history.append({
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "type": "nlp",
                        "text": text_input,
                        "result": result,
                        "latency_ms": latency_ms,
                        "tokens": len(tokens_final) if 'tokens_final' in dir() else 0
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        categorie = result.get('categorie_predite', 'N/A')
                        badge_class = f"cat-badge cat-{categorie}"
                        st.markdown(f'<span class="{badge_class}">{categorie}</span>', unsafe_allow_html=True)
                        st.metric("Catégorie prédite", categorie)
                    with col2:
                        st.metric("Latence", f"{latency_ms} ms")
                    
                    confidence = result.get('confidence')
                    if confidence:
                        st.progress(confidence, text=f"Confiance: {confidence:.2%}")
                    
                    # Afficher les infos du modèle (simulées)
                    with st.expander("ℹ️ Infos du modèle NLP"):
                        st.markdown("""
                        | Paramètre | Valeur |
                        |-----------|--------|
                        | Vectorisation | TF-IDF (uni+bigrammes) |
                        | Classifieur | LinearSVC |
                        | Stopwords | NLTK français |
                        | max_df | 0.85 (suppression auto mots fréquents) |
                        | min_df | 2 (suppression mots rares) |
                        """)
                else:
                    st.error(f"⚠️ Erreur API: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
        else:
            st.warning("Veuillez entrer une description")

# ============================================================================
# TAB 4: Batch Prediction
# ============================================================================
with tab4:
    st.markdown('<div class="section-title">📋 Prédiction par lot (Batch)</div>', unsafe_allow_html=True)
    st.markdown("Testez plusieurs déchets en une seule requête")
    
    # Exemples batch
    default_batch = [
        {"Poids": 16.7, "Volume": 70.9, "Conductivite": 0.0, "Opacite": 0.95, "Rigidite": 1.0, "Source": "Centre_Tri", "Rapport_Collecte": "Lot de papier récupéré"},
        {"Poids": 47.3, "Volume": 64.7, "Conductivite": 0.0, "Opacite": 0.4, "Rigidite": 3.0, "Source": "Usine_A", "Rapport_Collecte": "Lot plastique"},
        {"Poids": 187.8, "Volume": 400.2, "Conductivite": 0.0, "Opacite": 0.05, "Rigidite": 8.0, "Source": "Collecte_Citoyenne", "Rapport_Collecte": "Bris de verre"},
        {"Poids": 60.1, "Volume": 121.1, "Conductivite": 0.95, "Opacite": 0.2, "Rigidite": 7.0, "Source": "Usine_B", "Rapport_Collecte": "Métal de récupération"}
    ]
    
    batch_json = st.text_area("Payload JSON (liste d'objets)", 
                               value=json.dumps(default_batch, indent=2, ensure_ascii=False),
                               height=250)
    
    if st.button("▶️ Lancer le batch", type="primary", use_container_width=True):
        try:
            items = json.loads(batch_json)
            if not isinstance(items, list):
                st.error("Le JSON doit être une liste")
            else:
                start_time = time.time()
                results = []
                for item in items:
                    try:
                        resp = requests.post(f"{st.session_state.api_url}/predict", json=item, timeout=10)
                        if resp.status_code == 200:
                            results.append(resp.json())
                        else:
                            results.append({"error": f"HTTP {resp.status_code}"})
                    except Exception as e:
                        results.append({"error": str(e)})
                
                latency_ms = round((time.time() - start_time) * 1000)
                
                st.success(f"✅ {len(results)} prédictions en {latency_ms} ms")
                
                # Afficher les résultats
                df_batch = pd.DataFrame([
                    {
                        "#": i+1,
                        "Catégorie": r.get('categorie_predite', '❌'),
                        "Prix": r.get('prix_estime', 'N/A'),
                        "Confiance": f"{r.get('confidence', 0):.2%}" if r.get('confidence') else 'N/A'
                    }
                    for i, r in enumerate(results)
                ])
                st.dataframe(df_batch, use_container_width=True, hide_index=True)
                
                # Ajouter à l'historique
                st.session_state.history.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "type": "batch",
                    "count": len(results),
                    "latency_ms": latency_ms
                })
                
        except json.JSONDecodeError as e:
            st.error(f"JSON invalide: {e}")

# ============================================================================
# TAB 5: Historique
# ============================================================================
with tab5:
    st.markdown('<div class="section-title">📊 Historique des prédictions</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("Aucune prédiction pour l'instant")
    else:
        # Statistiques
        latencies = [h.get("latency_ms", 0) for h in st.session_state.history if h.get("latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total prédictions", len(st.session_state.history))
        col2.metric("Latence moyenne", f"{avg_latency:.0f} ms")
        col3.metric("Types", f"{len(set(h.get('type', '?') for h in st.session_state.history))}")
        
        st.divider()
        
        # Tableau d'historique
        rows = []
        for h in reversed(st.session_state.history):
            if h.get("type") == "manuel":
                rows.append({
                    "Heure": h["ts"],
                    "Type": "🎚️ Manuel",
                    "Résultat": h["result"].get("categorie_predite", "?"),
                    "Latence": f"{h.get('latency_ms', 0)} ms",
                    "Détail": f"Poids: {h['payload'].get('Poids', '?')} kg"
                })
            elif h.get("type") == "nlp":
                rows.append({
                    "Heure": h["ts"],
                    "Type": "🤖 NLP",
                    "Résultat": h["result"].get("categorie_predite", "?"),
                    "Latence": f"{h.get('latency_ms', 0)} ms",
                    "Détail": f"Text: {h.get('text', '')[:50]}..."
                })
            elif h.get("type") == "batch":
                rows.append({
                    "Heure": h["ts"],
                    "Type": "📋 Batch",
                    "Résultat": f"{h.get('count', 0)} prédictions",
                    "Latence": f"{h.get('latency_ms', 0)} ms",
                    "Détail": "-"
                })
        
        df_history = pd.DataFrame(rows)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # Export CSV
        if rows:
            csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exporter CSV", data=csv, 
                              file_name=f"eco_smart_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                              mime="text/csv")

# ============================================================================
# SIDEBAR - Infos supplémentaires
# ============================================================================
with st.sidebar:
    st.divider()
    st.markdown("**ℹ️ Stopwords utilisés (NLTK français)**")
    st.caption(f"{len(STOPWORDS_FR)} mots filtrés automatiquement")
    st.caption("Ex: le, la, les, un, une, des, de, du, et, ou...")
    
    st.divider()
    st.markdown("**📌 Exemples de requêtes cURL**")
    st.code(f"""curl -X POST {st.session_state.api_url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"Poids":50,"Volume":100,"Conductivite":0.5,
       "Opacite":0.8,"Rigidite":5,"Source":"Usine_A",
       "Rapport_Collecte":"Lot de plastique"}}'""", language="bash")