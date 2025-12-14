import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import base64
import os

# ================================================================
# 0. FONCTION IMAGE DE FOND (ROBUSTE ET DÉBOGUÉE)
# ================================================================
def get_base64_of_bin_file(bin_file):
    """Lit l'image pour l'encoder en base64 pour le CSS"""
    # --- DEBUG : Affiche le chemin exact où Python cherche l'image ---
    print(f"DEBUG: Tentative d'ouverture de l'image au chemin : {bin_file}")
    # -----------------------------------------------------------------
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        print(f"SUCCÈS: Image trouvée et lue ({len(data)} bytes)")
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        print("ERREUR: Le fichier n'a pas été trouvé à cet emplacement.")
        return None

# ================================================================
# 1. CONFIGURATION & STYLE CSS RENFORCÉ
# ================================================================

st.set_page_config(
    page_title="BankGuard AI | Détection Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CHARGEMENT IMAGE (Chemin Absolu) ---
# Trouve le dossier où se trouve ce fichier app.py
image_path = "background.jpg"


# Charge l'image
img_base64 = get_base64_of_bin_file(image_path)


# --- CSS RENFORCÉ ---
if img_base64:
    # Utilisation de .stApp::before pour le fond
    css_bg = f"""
    <style>
        /* Force le conteneur principal à être transparent */
        .stApp {{
            background-color: transparent !important;
        }}

        /* Crée la couche de fond floutée */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            /* Utilisation de l'image encodée */
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            /* Flou et transparence */
            filter: blur(10px);
            opacity: 0.9; 
            z-index: -1; /* Place le fond derrière tout le reste */
        }}
        
        /* Style des cartes (conteneurs blancs semi-transparents) */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
            background-color: rgba(255, 255, 255, 0.90) !important;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
    </style>
    """
else:
    # Fallback si l'image n'est pas trouvée
    css_bg = "<style>.stApp { background-color: #f0f2f6; }</style>"
    st.toast("⚠️ Image background.png introuvable. Regardez le terminal.", icon="🛑")

# Injection CSS
st.markdown(css_bg, unsafe_allow_html=True)

# Styles additionnels
st.markdown("""
<style>
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #1e293b; }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none; padding: 0.6rem 1rem;
        border-radius: 8px; font-weight: 600; width: 100%;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
    [data-testid="stSidebar"] { background-color: rgba(15, 23, 42, 0.95) !important; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 2. LOGIQUE MÉTIER & CHARGEMENT
# ================================================================

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model_lof_optimized.pkl")
        feature_path = os.path.join(current_dir, "feature_info.pkl")
        pipeline = joblib.load(model_path)
        features = joblib.load(feature_path)
        return pipeline, features, False
    except FileNotFoundError:
        return None, None, True

pipeline_lof, feature_info, DEMO_MODE = load_model()

def create_features(data):
    """Crée TOUTES les features requises"""
    df = pd.DataFrame([data])
    # Features de base
    df['ratio_to_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['ratio_to_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)
    df['error_orig'] = np.abs((df['oldbalanceOrg'] - df['newbalanceOrig']) - df['amount'])
    df['error_dest'] = np.abs((df['newbalanceDest'] - df['oldbalanceDest']) - df['amount'])
    df['error_orig_norm'] = df['error_orig'] / (df['amount'] + 1)
    df['error_dest_norm'] = df['error_dest'] / (df['amount'] + 1)
    df['orig_emptied'] = int((df['newbalanceOrig'].iloc[0] == 0) and (df['oldbalanceOrg'].iloc[0] > 0))
    df['dest_was_zero'] = int(df['oldbalanceDest'].iloc[0] == 0)
    # Features mathématiques requises (logs, deltas)
    df['amount_very_high'] = 0 
    df['amount_log'] = np.log1p(df['amount'])
    df['oldbalanceOrg_log'] = np.log1p(df['oldbalanceOrg'])
    df['oldbalanceDest_log'] = np.log1p(df['oldbalanceDest'])
    df['delta_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['delta_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    # Features temporelles
    df['hour'] = df['step'] % 24
    df['is_night'] = int((df['hour'].iloc[0] >= 22) or (df['hour'].iloc[0] <= 6))
    df['day_of_week'] = (df['step'] // 24) % 7
    df['is_weekend'] = int(df['day_of_week'].iloc[0] >= 5)
    return df

def calculate_balances(type_txn, amount, old_org, old_dest):
    new_org, new_dest = old_org, old_dest
    if type_txn == "PAYMENT": new_org = max(0, old_org - amount)
    elif type_txn == "TRANSFER": new_org = max(0, old_org - amount); new_dest = old_dest + amount
    elif type_txn == "CASH_OUT": new_org = max(0, old_org - amount)
    elif type_txn == "CASH_IN": new_org = old_org + amount
    elif type_txn == "DEBIT": new_org = max(0, old_org - amount)
    return new_org, new_dest

# ================================================================
# 3. INTERFACE UTILISATEUR
# ================================================================

with st.sidebar:
    st.title("🛡️ BankGuard ")
    st.markdown("---")
    if DEMO_MODE: st.warning(" Mode Simulation")
    else: st.success(" Modèle Connecté")
    auto_calc = st.checkbox("Calcul Solde Auto", value=True)

col_header = st.container()
with col_header:
    # Titres avec ombre pour lisibilité sur fond d'image
    st.markdown("<h1 style='text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>Analyse de Transaction</h1>", unsafe_allow_html=True)
    
col_inputs, col_results = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.subheader("Transaction")
    with st.container():
        c1, c2 = st.columns(2)
        type_txn = c1.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = c2.number_input("Montant (MAD)", 0.0, value=15000.0, step=100.0)
        c3, c4 = st.columns(2)
        day = c3.number_input("Jour", 1, 31, 1)
        hour = c4.slider("Heure", 0, 23, 14)

    st.subheader("Soldes")
    with st.container():
        old_org = st.number_input("Solde Origine (Avant)", 0.0, value=50000.0)
        old_dest = st.number_input("Solde Dest. (Avant)", 0.0, value=20000.0)
        if auto_calc:
            new_org, new_dest = calculate_balances(type_txn, amount, old_org, old_dest)
            st.caption(f" Nouveaux soldes : Origine {new_org:,.0f} | Dest {new_dest:,.0f}")
        else:
            new_org = st.number_input("Nouveau Solde Origine", 0.0, value=old_org)
            new_dest = st.number_input("Nouveau Solde Dest.", 0.0, value=old_dest)

    st.write("") 
    analyze_btn = st.button("Lancer l'Analyse ", use_container_width=True)

if analyze_btn:
    step = (day - 1) * 24 + hour
    txn_data = {
        'step': step, 'type': type_txn, 'amount': amount,
        'oldbalanceOrg': old_org, 'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest, 'newbalanceDest': new_dest
    }
    df_feat = create_features(txn_data)
    score, pred = 0, 1
    if not DEMO_MODE:
        try:
            X_input = df_feat[feature_info['all_features']]
            if 'preprocess' in pipeline_lof.named_steps:
                X_trans = pipeline_lof.named_steps['preprocess'].transform(X_input)
                if 'pca' in pipeline_lof.named_steps: X_trans = pipeline_lof.named_steps['pca'].transform(X_trans)
                score = pipeline_lof.named_steps['model'].score_samples(X_trans)[0]
            else: score = pipeline_lof.score_samples(X_input)[0]
            pred = -1 if score < -1.5 else 1 
        except Exception as e:
            st.error(f"Erreur technique : {e}")
            score, pred = -5, -1
    else:
        time.sleep(1)
        pred = -1 if (amount > 50000 and type_txn=="TRANSFER") else 1
        score = -2.5 if pred == -1 else -0.5

    with col_results:
        st.subheader("Résultats")
        if pred == -1:
            st.markdown(f"""<div style="background-color: #fee2e2; padding: 20px; border-radius: 10px; border-left: 6px solid #ef4444;"><h3 style="color: #b91c1c; margin:0;">🚨 FRAUDE SUSPECTÉE</h3><p>Score : <strong>{score:.2f}</strong></p></div>""", unsafe_allow_html=True)
            color = "#ef4444"
        else:
            st.markdown(f"""<div style="background-color: #dcfce7; padding: 20px; border-radius: 10px; border-left: 6px solid #22c55e;"><h3 style="color: #15803d; margin:0;"> TRANSACTION NORMALE</h3><p>Score : <strong>{score:.2f}</strong></p></div>""", unsafe_allow_html=True)
            color = "#22c55e"

        fig = go.Figure(go.Indicator(mode = "gauge+number", value = abs(score), title = {'text': "Niveau de Risque"}, gauge = {'axis': {'range': [0, 4]}, 'bar': {'color': color}, 'steps': [{'range': [0, 1.5], 'color': '#f0fdf4'}, {'range': [1.5, 4], 'color': '#fef2f2'}]}))
        fig.update_layout(height=250, margin=dict(t=40,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        vals = [min(1, amount/100000), 1 if df_feat['is_night'].iloc[0] else 0.1, min(1, df_feat['error_orig_norm'].iloc[0]*10), min(1, df_feat['ratio_to_orig'].iloc[0])]
        fig_radar = px.line_polar(r=vals, theta=['Montant', 'Nuit', 'Erreur Balance', 'Ratio'], line_close=True)
        fig_radar.update_traces(fill='toself', line_color=color)
        fig_radar.update_layout(height=200, margin=dict(t=20,b=20,l=40,r=40), polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

elif not analyze_btn:
    with col_results: st.info(" Entrez les données pour lancer l'analyse.")