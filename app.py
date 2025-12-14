import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import time
import base64
import os

# ================================================================
# 0. FONCTION IMAGE DE FOND (CLOUD-SAFE)
# ================================================================
def get_base64_of_bin_file(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

# ================================================================
# 1. CONFIGURATION PAGE
# ================================================================
st.set_page_config(
    page_title="BankGuard AI | Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, "background.jpg")
img_base64 = get_base64_of_bin_file(IMG_PATH)

# ================================================================
# 2. CSS GLOBAL
# ================================================================
if img_base64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            filter: blur(10px);
            opacity: 0.9;
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
<style>
h1, h2, h3 { font-family: Segoe UI, sans-serif; }
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#1d4ed8);
    color:white;
    border-radius:8px;
    font-weight:600;
    width:100%;
}
[data-testid="stSidebar"] {
    background-color: rgba(15,23,42,0.95);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# 3. CHARGEMENT MODÈLE (ROBUSTE)
# ================================================================
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(BASE_DIR, "model_lof_optimized.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "feature_info.pkl"), "rb") as f:
            features = pickle.load(f)
        return model, features, False
    except Exception:
        return None, None, True

pipeline_lof, feature_info, DEMO_MODE = load_model()

# ================================================================
# 4. FEATURE ENGINEERING
# ================================================================
def create_features(data):
    df = pd.DataFrame([data])

    df["ratio_to_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["ratio_to_dest"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["error_orig"] = abs((df["oldbalanceOrg"] - df["newbalanceOrig"]) - df["amount"])
    df["error_dest"] = abs((df["newbalanceDest"] - df["oldbalanceDest"]) - df["amount"])
    df["error_orig_norm"] = df["error_orig"] / (df["amount"] + 1)
    df["error_dest_norm"] = df["error_dest"] / (df["amount"] + 1)

    df["amount_log"] = np.log1p(df["amount"])
    df["oldbalanceOrg_log"] = np.log1p(df["oldbalanceOrg"])
    df["oldbalanceDest_log"] = np.log1p(df["oldbalanceDest"])

    df["delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["hour"] = df["step"] % 24
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["day_of_week"] = (df["step"] // 24) % 7
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["amount_very_high"] = 0
    df["orig_emptied"] = int(df["newbalanceOrig"].iloc[0] == 0)
    df["dest_was_zero"] = int(df["oldbalanceDest"].iloc[0] == 0)

    return df

def calculate_balances(tx, amount, old_org, old_dest):
    new_org, new_dest = old_org, old_dest
    if tx in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]:
        new_org = max(0, old_org - amount)
    if tx == "TRANSFER":
        new_dest += amount
    if tx == "CASH_IN":
        new_org += amount
    return new_org, new_dest

# ================================================================
# 5. INTERFACE
# ================================================================
with st.sidebar:
    st.title("🛡️ BankGuard AI")
    st.markdown("---")
    st.success("Modèle chargé" if not DEMO_MODE else "Mode démo")
    auto_calc = st.checkbox("Calcul automatique des soldes", True)

st.title("Analyse de Transaction Bancaire")

col_in, col_out = st.columns([1, 1.2])

with col_in:
    tx_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Montant", 0.0, value=15000.0, step=100.0)
    day = st.number_input("Jour", 1, 7, 1)
    hour = st.slider("Heure", 0, 23, 14)

    old_org = st.number_input("Solde origine", 0.0, value=50000.0)
    old_dest = st.number_input("Solde destination", 0.0, value=20000.0)

    if auto_calc:
        new_org, new_dest = calculate_balances(tx_type, amount, old_org, old_dest)
    else:
        new_org = st.number_input("Nouveau solde origine", 0.0)
        new_dest = st.number_input("Nouveau solde destination", 0.0)

    analyze = st.button("Analyser la transaction")

# ================================================================
# 6. PRÉDICTION & VISUALISATION
# ================================================================
if analyze:
    step = (day - 1) * 24 + hour

    data = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    }

    df_feat = create_features(data)

    if DEMO_MODE:
        score = -2.5 if amount > 50000 else -0.3
        pred = -1 if score < -1 else 1
    else:
        X = df_feat[feature_info["all_features"]]
        score = pipeline_lof.score_samples(X)[0]
        pred = -1 if score < -1.5 else 1

    with col_out:
        if pred == -1:
            st.error(f"🚨 Fraude suspectée — Score {score:.2f}")
        else:
            st.success(f" Transaction normale — Score {score:.2f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(score),
            gauge={"axis": {"range": [0, 4]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

else:
    with col_out:
        st.info("Veuillez entrer les données et lancer l’analyse.")
