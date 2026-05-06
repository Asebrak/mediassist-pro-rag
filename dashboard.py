"""
dashboard.py — MediAssist Pro
Interface Streamlit 2026 — Glassmorphism + Ubuntu font
Connecté à RAGEngine (index FAISS réel + Groq)
"""

import streamlit as st
import os, json, time
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go
from collections import Counter

# ─── Import du moteur RAG central ─────────────────────────────────────────────
from rag_engine import RAGEngine

load_dotenv()

# ───────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediAssist Pro",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────────────────────────────────────────────────────────────
#  CSS MODERNE 2026 — Glassmorphism + Ubuntu
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Ubuntu+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: #080e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Ubuntu', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stAppViewContainer"] { background: transparent !important; }
section[data-testid="stSidebar"]  { display: none !important; }

[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,.08) 0%, transparent 60%),
        #080e1a;
    pointer-events: none; z-index: 0;
}

[data-testid="stMainBlockContainer"] {
    position: relative; z-index: 1;
    padding: 0 !important; max-width: 100% !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 999px; }

/* ── Glass ── */
.glass {
    background: rgba(15,23,42,.55);
    backdrop-filter: blur(20px) saturate(150%);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px; padding: 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.05);
    transition: all .3s cubic-bezier(.4,0,.2,1);
    margin-bottom: 16px;
}
.glass:hover {
    border-color: rgba(99,102,241,.2);
    box-shadow: 0 16px 48px rgba(99,102,241,.12), inset 0 1px 0 rgba(255,255,255,.07);
    transform: translateY(-2px);
}

/* ── Header ── */
.header-wrap {
    background: rgba(8,14,26,.85);
    backdrop-filter: blur(30px);
    border-bottom: 1px solid rgba(255,255,255,.06);
    padding: 14px 36px;
    display: flex; align-items: center; justify-content: space-between;
}
.logo-text {
    font-weight: 700; font-size: 21px; letter-spacing: -.5px;
    background: linear-gradient(135deg,#818cf8,#c084fc,#67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hbadge {
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 999px; padding: 4px 13px;
    font-size: 11px; color: #a5b4fc; font-weight: 500; letter-spacing: .4px;
    display: inline-block; margin-left: 8px;
}
.hbadge-green {
    background: rgba(16,185,129,.1);
    border-color: rgba(16,185,129,.3); color: #6ee7b7;
}

/* ── Section label ── */
.slabel {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #475569; margin-bottom: 12px;
}

/* ── Profile ── */
.p-avatar {
    width: 48px; height: 48px; border-radius: 14px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(99,102,241,.4);
}
.p-name { font-size: 17px; font-weight: 700; color: #f1f5f9; letter-spacing: -.3px; }
.p-sub  { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.p-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 7px; padding: 4px 9px;
    font-size: 11px; color: #94a3b8; margin: 2px;
}

/* ── Security badges ── */
.badge-vert   { display:inline-flex;align-items:center;gap:8px;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);border-radius:12px;padding:10px 16px;color:#6ee7b7;font-weight:600;font-size:14px;box-shadow:0 0 20px rgba(16,185,129,.15); }
.badge-orange { display:inline-flex;align-items:center;gap:8px;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);border-radius:12px;padding:10px 16px;color:#fcd34d;font-weight:600;font-size:14px;box-shadow:0 0 20px rgba(245,158,11,.15); }
.badge-rouge  { display:inline-flex;align-items:center;gap:8px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);border-radius:12px;padding:10px 16px;color:#fca5a5;font-weight:600;font-size:14px;animation:pred 2s ease-in-out infinite; }
@keyframes pred { 0%,100%{box-shadow:0 0 20px rgba(239,68,68,.2)}50%{box-shadow:0 0 36px rgba(239,68,68,.4)} }

/* ── Med card ── */
.med-card {
    background: rgba(15,23,42,.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 14px; padding: 18px 20px; margin: 8px 0;
    position: relative; overflow: hidden;
    transition: all .25s ease;
}
.med-card::before {
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4);
}
.med-card:hover { border-color:rgba(99,102,241,.2); transform:translateX(3px); }
.med-name { font-size:15px;font-weight:700;color:#e2e8f0;letter-spacing:-.2px; }
.sbar { height:4px;border-radius:999px;background:rgba(255,255,255,.07);margin:8px 0 4px;overflow:hidden; }
.sfill { height:100%;border-radius:999px;background:linear-gradient(90deg,#6366f1,#06b6d4); }

/* ── Bubbles ── */
.bub-user {
    background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(139,92,246,.12));
    border:1px solid rgba(99,102,241,.22);
    border-radius:16px 16px 4px 16px; padding:13px 17px;
    max-width:72%; margin-left:auto; font-size:14px; color:#e2e8f0;
}
.bub-ai {
    background:rgba(15,23,42,.65);
    border:1px solid rgba(255,255,255,.06);
    border-radius:16px 16px 16px 4px; padding:16px 18px;
    font-size:14px; line-height:1.75; color:#cbd5e1;
}
.bub-time { font-size:10px;color:#475569;margin-top:4px;font-family:'Ubuntu Mono',monospace; }

/* ── Timeline ── */
.tl-item {
    border-left:2px solid rgba(99,102,241,.2);
    padding-left:16px; margin-left:8px; margin-bottom:18px; position:relative;
}
.tl-item::before {
    content:''; width:9px;height:9px; border-radius:50%;
    background:#6366f1; position:absolute; left:-5px; top:4px;
    box-shadow:0 0 8px rgba(99,102,241,.5);
}
.tl-q    { font-size:13px;font-weight:600;color:#e2e8f0; }
.tl-meta { font-size:11px;color:#475569;margin-top:2px;font-family:'Ubuntu Mono',monospace; }

/* ── Metric pill ── */
.mpill {
    display:inline-flex;align-items:center;gap:5px;
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.06);
    border-radius:9px; padding:6px 12px;
    font-size:12px;color:#94a3b8; margin:3px;
}
.mpill-val { font-weight:700;color:#c7d2fe;font-family:'Ubuntu Mono',monospace; }

/* ── Source chip ── */
.schip {
    display:inline-flex;align-items:center;gap:5px;
    background:rgba(6,182,212,.09);border:1px solid rgba(6,182,212,.22);
    border-radius:7px;padding:4px 11px;font-size:11px;color:#67e8f9;margin:3px;
}

/* ── Fav item ── */
.fav-item {
    display:flex;align-items:center;justify-content:space-between;
    background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.05);
    border-radius:11px;padding:10px 14px;margin:5px 0;transition:all .2s ease;
}
.fav-item:hover { background:rgba(99,102,241,.07);border-color:rgba(99,102,241,.18); }

/* ── Alert ── */
.alert-danger { background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.28);border-radius:11px;padding:12px 16px;color:#fca5a5;font-size:13px; }
.alert-info   { background:rgba(99,102,241,.09);border:1px solid rgba(99,102,241,.22);border-radius:11px;padding:12px 16px;color:#a5b4fc;font-size:13px; }
.glow-div { height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.35),transparent);border:none;margin:22px 0; }

/* ── Streamlit overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background:rgba(15,23,42,.7) !important;
    border:1px solid rgba(255,255,255,.09) !important;
    border-radius:13px !important; color:#e2e8f0 !important;
    font-family:'Ubuntu',sans-serif !important; font-size:14px !important;
    padding:12px 16px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color:rgba(99,102,241,.45) !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.09) !important;
}
[data-testid="stButton"] button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; border-radius:11px !important;
    color:white !important; font-family:'Ubuntu',sans-serif !important;
    font-weight:600 !important; font-size:13px !important;
    transition:all .25s ease !important;
    box-shadow:0 4px 14px rgba(99,102,241,.32) !important;
}
[data-testid="stButton"] button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 22px rgba(99,102,241,.42) !important;
}
[data-testid="stSelectbox"] > div > div {
    background:rgba(15,23,42,.7) !important;
    border:1px solid rgba(255,255,255,.09) !important;
    border-radius:11px !important; color:#e2e8f0 !important;
}
[data-testid="stExpander"] {
    background:rgba(15,23,42,.5) !important;
    border:1px solid rgba(255,255,255,.06) !important;
    border-radius:13px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family:'Ubuntu',sans-serif !important; color:#64748b !important;
    font-weight:500 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#818cf8 !important; border-bottom-color:#6366f1 !important;
}
hr { border-color:rgba(255,255,255,.05) !important; }
a  { color:#818cf8 !important; text-decoration:none !important; }
a:hover { color:#c084fc !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ───────────────────────────────────────────────────────────────────────────────

for k, v in {
    "profil":     None,
    "historique": [],
    "favoris":    {},
    "page":       "onboarding",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────────────────────────────────────────────────────────
#  CHARGER LE MOTEUR RAG (cached — chargé UNE SEULE FOIS)
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine():
    """Charge RAGEngine une seule fois (index FAISS + modèle + Groq)."""
    try:
        return RAGEngine(), True
    except Exception as e:
        return str(e), False

# ───────────────────────────────────────────────────────────────────────────────
#  HELPER UI
# ───────────────────────────────────────────────────────────────────────────────

def render_header():
    profil = st.session_state.profil
    prenom = profil.get("prenom","") if profil else ""
    heure  = datetime.now().strftime("%H:%M")
    st.markdown(f"""
    <div class="header-wrap">
        <div style="display:flex;align-items:center;gap:14px">
            <span style="font-size:26px">💊</span>
            <div>
                <div class="logo-text">MediAssist Pro</div>
                <div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase">
                    BDPM/ANSM · Groq LLaMA 3.3 · FAISS
                </div>
            </div>
        </div>
        <div>
            {"<span class='hbadge'>👤 "+prenom+"</span>" if prenom else ""}
            <span class="hbadge">🕐 {heure}</span>
            <span class="hbadge hbadge-green">● Live</span>
        </div>
    </div>
    <div style="height:20px"></div>
    """, unsafe_allow_html=True)


def render_score(score: dict):
    n   = score.get("niveau","ORANGE")
    css = {"VERT":"badge-vert","ORANGE":"badge-orange","ROUGE":"badge-rouge"}[n]
    ico = {"VERT":"🟢","ORANGE":"🟡","ROUGE":"🔴"}[n]
    st.markdown(f'<div class="{css}">{ico} Sécurité : {n}</div>', unsafe_allow_html=True)
    for r in score.get("raisons",[]):
        st.markdown(f'<div class="mpill">→ {r}</div>', unsafe_allow_html=True)
    ci = [x for x in score.get("contre_indications_detectees",[]) if x]
    if ci:
        st.markdown(f'<div class="alert-danger">⛔ Contre-indication : {", ".join(ci)}</div>', unsafe_allow_html=True)


def render_chunks(chunks: list):
    for c in chunks:
        denom = c["metadata"].get("denomination","?")
        pct   = int(c["score"] * 100)
        url   = c["metadata"].get("url_prospectus","")
        st.markdown(f"""
        <div class="med-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <div class="med-name">💊 {denom[:55]}</div>
                <div class="mpill"><span class="mpill-val">{pct}%</span> pertinence</div>
            </div>
            <div class="sbar"><div class="sfill" style="width:{pct}%"></div></div>
            <div style="font-size:11px;color:#64748b;margin-top:6px">{c['contenu'][:130].strip()}…</div>
            {"<div style='margin-top:10px'><a href='"+url+"' target='_blank' class='schip'>🔗 Prospectus officiel BDPM/ANSM</a></div>" if url else ""}
        </div>
        """, unsafe_allow_html=True)


def render_block(item: dict):
    """Affiche un bloc question + réponse + score + trace."""
    ts = item.get("timestamp","")
    st.markdown(f'<div class="bub-user">{item["question"]}<div class="bub-time">{ts}</div></div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if item.get("score_secu"):
        render_score(item["score_secu"])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if item.get("reponse"):
        st.markdown(f'<div class="bub-ai">{item["reponse"]}</div>', unsafe_allow_html=True)

    if item.get("chunks"):
        with st.expander("💊 Médicaments sources dans la base FAISS"):
            render_chunks(item["chunks"])

    if item.get("trace"):
        tr = item["trace"]
        h  = tr.get("hallucinations",{})
        with st.expander("🔍 Trace complète — Explainability"):
            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:4px">
                <div class="mpill">🔍 Reformulation : <span class="mpill-val">{tr.get('query_reformulee','')}</span></div>
                <div class="mpill">📦 Chunks retenus : <span class="mpill-val">{tr.get('chunks_trouves',0)}</span></div>
                <div class="mpill">🎯 Meilleur score : <span class="mpill-val">{tr.get('best_score',0):.2f}</span></div>
                <div class="mpill">🧠 Tokens contexte : <span class="mpill-val">{tr.get('tokens_contexte',0)}</span></div>
                <div class="mpill">⚡ Latence : <span class="mpill-val">{tr.get('latence','?')}</span></div>
                <div class="mpill">🧬 Hallucinations : <span class="mpill-val">{h.get('verdict','?')}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#  PAGE ONBOARDING
# ───────────────────────────────────────────────────────────────────────────────

def page_onboarding():
    st.markdown("""
    <div style="text-align:center;padding:50px 20px 30px">
        <div style="font-size:64px;animation:float 3s ease-in-out infinite">💊</div>
        <div style="font-family:Ubuntu;font-size:40px;font-weight:700;
            background:linear-gradient(135deg,#818cf8,#c084fc,#67e8f9);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;line-height:1.1;margin:12px 0 8px">
            MediAssist Pro
        </div>
        <div style="color:#475569;font-size:15px;font-weight:300;letter-spacing:.3px">
            Assistant pharmaceutique personnalisé — Données BDPM / ANSM officielles
        </div>
    </div>
    <style>
    @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="glass" style="padding:28px">', unsafe_allow_html=True)
        st.markdown('<p class="slabel">👤 Ton profil médical</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748b;font-size:13px;margin-bottom:18px">Ces informations permettent d\'adapter chaque réponse à ta situation personnelle.</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            prenom    = st.text_input("😊 Prénom", placeholder="Ahmed")
            age       = st.text_input("🎂 Âge", placeholder="24")
            sexe      = st.selectbox("🧬 Sexe", ["—", "Homme", "Femme", "Autre"])
        with c2:
            poids     = st.text_input("⚖️ Poids (kg)", placeholder="75")
            grossesse = st.selectbox("🤰 Grossesse ?", ["Non", "Oui", "Possible"])
            allergies = st.text_input("⚠️ Allergies", placeholder="pénicilline... ou aucune")

        medicaments = st.text_input("💊 Médicaments en cours", placeholder="metformine... ou aucun")
        antecedents = st.text_input("🏥 Antécédents médicaux", placeholder="ulcère, diabète... ou aucun")
        symptomes   = st.text_area("🤒 Qu'est-ce qui t'amène ?", placeholder="Décris ce que tu ressens...", height=75)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if st.button("✨ Démarrer ma consultation", use_container_width=True):
            if prenom:
                st.session_state.profil = {
                    "prenom": prenom, "age": age or "?", "sexe": sexe,
                    "poids": poids or "?", "grossesse": grossesse,
                    "allergies": allergies or "aucune",
                    "medicaments": medicaments or "aucun",
                    "antecedents": antecedents or "aucun",
                    "symptomes_now": symptomes or "Non renseigné",
                }
                if symptomes.strip():
                    st.session_state.historique.append({
                        "question":  symptomes.strip(),
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "type":      "symptomes",
                        "pending":   True,
                    })
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Entre au moins ton prénom pour commencer.")

# ───────────────────────────────────────────────────────────────────────────────
#  PAGE DASHBOARD
# ───────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    render_header()

    # ── Charger le moteur RAG ─────────────────────────────────────────────────
    result = get_engine()
    engine, ok = result
    if not ok:
        st.markdown(f'<div class="alert-danger">❌ {engine}<br>Lancez : python indexation.py</div>',
                    unsafe_allow_html=True)
        return

    profil = st.session_state.profil
    prenom = profil.get("prenom","")

    # ── Layout 3 colonnes ─────────────────────────────────────────────────────
    left, main, right = st.columns([1, 2.6, 1], gap="medium")

    # ─────────────────────────────────────────────────────────────────────────
    #  LEFT — Profil + Favoris
    # ─────────────────────────────────────────────────────────────────────────
    with left:
        sexe_ico = "👨" if "homme" in profil.get("sexe","").lower() else "👩" if "femme" in profil.get("sexe","").lower() else "🧑"
        st.markdown(f"""
        <div class="glass">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
                <div class="p-avatar">{sexe_ico}</div>
                <div>
                    <div class="p-name">{prenom}</div>
                    <div class="p-sub">{profil.get('age','?')} ans · {profil.get('poids','?')} kg</div>
                </div>
            </div>
            <span class="p-chip">🧬 {profil.get('sexe','?')}</span>
            <span class="p-chip">🤰 {profil.get('grossesse','?')}</span>
            <span class="p-chip">⚠️ {profil.get('allergies','?')[:18]}</span>
            <span class="p-chip">💊 {profil.get('medicaments','?')[:18]}</span>
            <span class="p-chip">🏥 {profil.get('antecedents','?')[:18]}</span>
            <div class="glow-div"></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✏️ Modifier profil", use_container_width=True):
            st.session_state.page   = "onboarding"
            st.session_state.profil = None
            st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Favoris
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<p class="slabel">❤️ Favoris sauvegardés</p>', unsafe_allow_html=True)
        if st.session_state.favoris:
            for nom, info in list(st.session_state.favoris.items())[:6]:
                st.markdown(f"""
                <div class="fav-item">
                    <div>
                        <div style="font-size:12px;font-weight:600;color:#e2e8f0">💊 {nom[:22]}</div>
                        <div style="font-size:10px;color:#475569">{info.get('date','')}</div>
                    </div>
                    <div style="font-size:10px;color:#64748b">{info.get('count',1)}×</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#475569;font-size:12px">Aucun favori.<br>Ils s\'ajoutent automatiquement après chaque réponse.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN — Tabs
    # ─────────────────────────────────────────────────────────────────────────
    with main:
        tab_chat, tab_compare, tab_history, tab_edu = st.tabs([
            "💬 Consultation", "🔬 Comparer", "📅 Historique", "📚 Mode Éducation"
        ])

        # ── TAB 1 : Chat ──────────────────────────────────────────────────────
        with tab_chat:

            # Afficher les échanges passés
            for item in st.session_state.historique:
                if not item.get("pending"):
                    render_block(item)

            # Traiter les pending
            pending = [i for i in st.session_state.historique if i.get("pending")]
            if pending:
                item = pending[0]
                with st.spinner("🔍 Recherche dans la base FAISS · Groq génère la réponse..."):

                    if item.get("type") == "symptomes":
                        res = engine.mode_symptomes(item["question"], profil)
                    else:
                        hist_msgs = [
                            msg for i in st.session_state.historique
                            if not i.get("pending") and i.get("reponse")
                            for msg in [
                                {"role": "user",      "content": i["question"]},
                                {"role": "assistant", "content": i["reponse"]},
                            ]
                        ]
                        res = engine.pipeline(item["question"], profil, hist_msgs)

                    if res.get("succes"):
                        # Auto-favoris
                        for c in res.get("chunks", []):
                            denom = c["metadata"].get("denomination", "?")[:30]
                            if denom not in st.session_state.favoris:
                                st.session_state.favoris[denom] = {
                                    "date":  datetime.now().strftime("%d/%m %H:%M"),
                                    "count": 1,
                                }
                            else:
                                st.session_state.favoris[denom]["count"] += 1

                        item.update({
                            "reponse":    res.get("reponse",""),
                            "score_secu": res.get("score_secu"),
                            "chunks":     res.get("chunks",[]),
                            "trace":      res.get("trace",{}),
                        })
                    else:
                        item["reponse"] = res.get("message","Aucune information trouvée.")

                item["pending"] = False
                st.rerun()

            st.markdown('<hr class="glow-div">', unsafe_allow_html=True)

            # Input
            st.markdown('<p class="slabel">✍️ Nouvelle question</p>', unsafe_allow_html=True)
            question = st.text_area(
                label="", label_visibility="collapsed",
                placeholder=f"Pose ta question, {prenom}... (ex: effets secondaires ibuprofène, puis-je prendre X avec Y ?)",
                height=85, key="q_input"
            )

            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                if st.button("🚀 Envoyer", use_container_width=True):
                    if question.strip():
                        st.session_state.historique.append({
                            "question":  question.strip(),
                            "timestamp": datetime.now().strftime("%H:%M"),
                            "type":      "question",
                            "pending":   True,
                        })
                        st.rerun()
            with c2:
                if st.button("🎯 Symptômes", use_container_width=True):
                    if question.strip():
                        st.session_state.historique.append({
                            "question":  question.strip(),
                            "timestamp": datetime.now().strftime("%H:%M"),
                            "type":      "symptomes",
                            "pending":   True,
                        })
                        st.rerun()
            with c3:
                if st.button("🗑️ Effacer", use_container_width=True):
                    st.session_state.historique = []
                    st.rerun()

            st.markdown('<p style="color:#334155;font-size:11px;text-align:center;margin-top:8px">⚠️ Ces informations ne remplacent pas l\'avis d\'un professionnel de santé · BDPM/ANSM</p>', unsafe_allow_html=True)

        # ── TAB 2 : Comparaison ───────────────────────────────────────────────
        with tab_compare:
            st.markdown('<p class="slabel">🔬 Comparaison de deux médicaments</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                med1 = st.text_input("💊 Médicament 1", placeholder="ibuprofène", key="med1")
            with c2:
                med2 = st.text_input("💊 Médicament 2", placeholder="paracétamol", key="med2")

            if st.button("🔬 Lancer la comparaison", use_container_width=True):
                if med1 and med2:
                    with st.spinner("Comparaison en cours via FAISS + Groq..."):
                        res = engine.comparer(med1, med2, profil)

                    # Radar chart
                    cats = ["Efficacité", "Tolérance digestive", "Sécurité", "Rapidité", "Simplicité posologie"]
                    s1 = [min(res["chunks1"][0]["score"]*9,9) if res["chunks1"] else 5, 6, 7, 7, 6]
                    s2 = [min(res["chunks2"][0]["score"]*9,9) if res["chunks2"] else 5, 8, 8, 6, 8]

                    fig = go.Figure()
                    for scores, nom, col, fill in [
                        (s1, med1, "#818cf8", "rgba(129,140,248,.15)"),
                        (s2, med2, "#34d399", "rgba(52,211,153,.15)"),
                    ]:
                        fig.add_trace(go.Scatterpolar(
                            r=scores+[scores[0]], theta=cats+[cats[0]],
                            fill="toself", name=nom,
                            line=dict(color=col, width=2),
                            fillcolor=fill,
                        ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0,10],
                                gridcolor="rgba(255,255,255,.07)", color="#475569"),
                            angularaxis=dict(gridcolor="rgba(255,255,255,.07)", color="#94a3b8"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Ubuntu", color="#cbd5e1", size=12),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                        margin=dict(t=20,b=20,l=40,r=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f'<div class="bub-ai">{res["reponse"]}</div>', unsafe_allow_html=True)
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                    render_chunks(res["chunks1"][:2] + res["chunks2"][:2])

        # ── TAB 3 : Historique ────────────────────────────────────────────────
        with tab_history:
            st.markdown('<p class="slabel">📅 Timeline de tes consultations</p>', unsafe_allow_html=True)
            done = [i for i in st.session_state.historique if not i.get("pending")]

            if not done:
                st.markdown('<div class="alert-info">Aucune consultation pour l\'instant. Pose ta première question dans l\'onglet Consultation !</div>', unsafe_allow_html=True)
            else:
                for item in reversed(done):
                    n   = item.get("score_secu",{}).get("niveau","?") if item.get("score_secu") else "?"
                    ico = {"VERT":"🟢","ORANGE":"🟡","ROUGE":"🔴"}.get(n,"⚪")
                    typ = "🎯 Symptômes" if item.get("type")=="symptomes" else "💬 Question"
                    st.markdown(f"""
                    <div class="tl-item">
                        <div style="display:flex;align-items:center;gap:8px">
                            <div class="tl-q">{item['question'][:75]}</div>
                            <div class="mpill">{ico} {n}</div>
                        </div>
                        <div class="tl-meta">{item.get('timestamp','')} · {typ}</div>
                    </div>
                    """, unsafe_allow_html=True)

                export = [
                    {"question": i["question"], "timestamp": i.get("timestamp",""),
                     "securite": i.get("score_secu",{}).get("niveau","?"),
                     "reponse":  i.get("reponse","")}
                    for i in done
                ]
                st.download_button(
                    label="📥 Exporter l'historique JSON",
                    data=json.dumps(export, ensure_ascii=False, indent=2),
                    file_name=f"mediassist_{prenom}_{datetime.now():%Y%m%d}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # ── TAB 4 : Mode Éducation ────────────────────────────────────────────
        with tab_edu:
            st.markdown('<p class="slabel">📚 Explication en langage simple</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#64748b;font-size:13px;margin-bottom:14px">Pose une question complexe — je t\'explique avec des analogies simples, sans jargon médical.</p>', unsafe_allow_html=True)

            q_edu = st.text_input("", placeholder="Ex: Pourquoi l'ibuprofène est dangereux pendant la grossesse ?", key="q_edu")
            if st.button("📚 Expliquer simplement", use_container_width=True):
                if q_edu:
                    with st.spinner("Génération de l'explication..."):
                        edu = engine.expliquer_simplement(q_edu, profil)
                    st.markdown(f'<div class="bub-ai">{edu}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  RIGHT — Métriques live
    # ─────────────────────────────────────────────────────────────────────────
    with right:
        done = [i for i in st.session_state.historique if not i.get("pending")]
        nb_v = sum(1 for i in done if i.get("score_secu",{}).get("niveau")=="VERT")
        nb_o = sum(1 for i in done if i.get("score_secu",{}).get("niveau")=="ORANGE")
        nb_r = sum(1 for i in done if i.get("score_secu",{}).get("niveau")=="ROUGE")

        st.markdown(f"""
        <div class="glass">
            <p class="slabel">📊 Métriques live</p>
            <div class="mpill">💬 Consultations : <span class="mpill-val">{len(done)}</span></div>
            <div class="mpill">🟢 Sécurité verte : <span class="mpill-val">{nb_v}</span></div>
            <div class="mpill">🟡 Précautions : <span class="mpill-val">{nb_o}</span></div>
            <div class="mpill">🔴 Alertes : <span class="mpill-val">{nb_r}</span></div>
            <div class="mpill">❤️ Favoris : <span class="mpill-val">{len(st.session_state.favoris)}</span></div>
            <div class="mpill">🗄️ Base FAISS : <span class="mpill-val">{engine.index.ntotal:,}</span></div>
            <div class="mpill">📦 Chunks : <span class="mpill-val">{len(engine.chunks_meta):,}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Dernière alerte
        alertes = [i for i in done if i.get("score_secu",{}).get("niveau") in ["ORANGE","ROUGE"]]
        if alertes:
            d = alertes[-1]
            n = d["score_secu"]["niveau"]
            st.markdown(f"""
            <div class="glass">
                <p class="slabel">⚠️ Dernière alerte</p>
                <div class="{'alert-danger' if n=='ROUGE' else 'alert-info'}">
                    {"🔴" if n=="ROUGE" else "🟡"} {n}<br>
                    <span style="font-size:11px;opacity:.8">{d['score_secu'].get('recommandation','')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recommandation intelligente
        if len(done) >= 2:
            mots = Counter(
                " ".join(i["question"] for i in done).lower().split()
            ).most_common(10)
            mots_filtres = [m for m, _ in mots if len(m) > 4 and m not in {"quels","sont","pour","avec","peut","prendre"}]
            if mots_filtres:
                st.markdown(f"""
                <div class="glass">
                    <p class="slabel">💡 Recommandation</p>
                    <p style="color:#94a3b8;font-size:12px;line-height:1.6">
                        Tu consultes souvent sur
                        <b style="color:#c7d2fe">{', '.join(mots_filtres[:2])}</b>.<br>
                        Veux-tu une explication approfondie dans l'onglet <b style="color:#818cf8">Mode Éducation</b> ?
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#  ROUTER
# ───────────────────────────────────────────────────────────────────────────────

if st.session_state.page == "onboarding" or not st.session_state.profil:
    page_onboarding()
else:
    page_dashboard()
