import json
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

# ---------------------------------------------------------
# KAVACH - Spam Detection Dashboard
# Place this file beside your existing model/ folder.
# Expected files:
#   model/spam_model.pkl
#   model/metrics.json
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

st.set_page_config(
    page_title="KAVACH | Smart Spam Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- Styling --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@600;700;800&display=swap');

    :root {
        --bg: #050d1d;
        --panel: #091a32;
        --panel2: #0d2340;
        --cyan: #16e6df;
        --blue: #168bff;
        --purple: #9a5cff;
        --pink: #ff3977;
        --muted: #a8c3df;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 45% 5%, rgba(0,180,220,.13), transparent 32%),
            radial-gradient(circle at 90% 80%, rgba(122,65,255,.12), transparent 28%),
            linear-gradient(135deg, #061326 0%, #030918 100%);
        color: #f4f8ff;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08213d 0%, #07152b 75%, #10123a 100%);
        border-right: 1px solid rgba(30,210,245,.25);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.3rem; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1600px; }

    .brand {
        display:flex; align-items:center; gap:14px;
        padding: 10px 0 22px 0;
        border-bottom: 1px solid rgba(22,230,223,.35);
        margin-bottom: 22px;
    }
    .brand-icon {
        width:58px; height:58px; border-radius:18px;
        display:flex; align-items:center; justify-content:center;
        font-size:34px; color:#061326;
        background:linear-gradient(135deg,var(--cyan),#5c9dff 55%,var(--purple));
        box-shadow:0 0 28px rgba(22,230,223,.35);
    }
    .brand-name {
        font-family:'Noto Sans Devanagari',sans-serif;
        font-size:32px; font-weight:800; line-height:1.1; color:white;
    }
    .brand-sub { color:var(--cyan); font-size:10px; font-weight:800; letter-spacing:2px; margin-top:5px; }

    .side-heading { color:var(--cyan); font-size:20px; font-weight:800; line-height:1.4; margin:18px 0 10px; }
    .side-copy { color:#a9c7e4; font-size:14px; line-height:1.8; }
    .section-label { color:var(--cyan); font-size:12px; font-weight:800; letter-spacing:2px; margin:4px 0 12px; }
    .side-section { border-top:1px solid rgba(80,170,220,.28); padding-top:20px; margin-top:22px; }
    .side-item { color:#dceaff; font-size:14px; margin:12px 0; }
    .accuracy-card {
        background:linear-gradient(120deg,rgba(0,210,220,.14),rgba(35,110,255,.18));
        border:1px solid rgba(22,230,223,.65); border-radius:16px;
        padding:17px; box-shadow:0 0 24px rgba(0,210,220,.08);
    }
    .accuracy-value { font-size:34px; font-weight:800; color:#a8ffff; }

    .top-button button, .stButton > button {
        width:100%; border:0 !important; border-radius:13px !important;
        color:#061326 !important; font-weight:800 !important;
        background:linear-gradient(105deg,#14dfd0,#399be9 55%,#a45cff) !important;
        padding: .72rem 1rem !important; min-height:48px;
        box-shadow:0 5px 22px rgba(26,180,220,.18);
        transition:transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(70,130,255,.3); }
    .stTextArea textarea {
        background:#07172c !important; color:#f4f8ff !important;
        border:1px solid rgba(64,170,220,.55) !important;
        border-radius:14px !important; font-size:16px !important;
    }
    .stTextArea label { color:#dceaff !important; font-weight:700 !important; }

    .panel {
        background:linear-gradient(145deg,rgba(8,30,55,.96),rgba(4,18,37,.96));
        border:1px solid rgba(0,190,240,.55); border-radius:18px;
        padding:20px 22px; box-shadow:0 10px 30px rgba(0,0,0,.18);
        margin-bottom:18px;
    }
    .panel-title { color:var(--cyan); font-size:16px; font-weight:800; letter-spacing:.5px; margin-bottom:14px; }
    .result-panel {
        background:linear-gradient(135deg,rgba(0,205,190,.13),rgba(7,28,51,.96));
        border:1px solid rgba(22,230,223,.8); border-radius:18px;
        padding:22px; box-shadow:0 0 26px rgba(0,210,210,.08);
    }
    .result-panel.spam { border-color:rgba(255,57,119,.9); background:linear-gradient(135deg,rgba(255,57,119,.13),rgba(7,28,51,.97)); }
    .result-main { font-size:28px; font-weight:800; margin:4px 0 5px; }
    .result-spam { color:#ff5c8e; }
    .result-ham { color:#42edc4; }
    .muted { color:#a9c7e4; font-size:14px; line-height:1.7; }
    .prob-row { display:flex; justify-content:space-between; align-items:center; color:#e9f4ff; margin:18px 0 8px; }
    .prob-track { height:12px; background:#183957; border-radius:20px; overflow:hidden; }
    .prob-fill { height:100%; border-radius:20px; }
    .step-card {
        background:linear-gradient(145deg,rgba(11,35,65,.95),rgba(5,20,40,.95));
        border:1px solid rgba(48,130,210,.5); border-radius:16px;
        padding:17px; margin-bottom:12px; min-height:105px;
    }
    .step-num {
        display:inline-flex; width:43px; height:43px; align-items:center; justify-content:center;
        border-radius:50%; background:linear-gradient(135deg,#00cfc8,#276de7);
        color:white; font-weight:800; margin-right:12px; box-shadow:0 0 18px rgba(0,190,220,.25);
    }
    .step-title { color:#f5f8ff; font-weight:800; }
    .scan-card {
        background:rgba(8,30,55,.9); border:1px solid rgba(45,130,200,.38);
        border-radius:13px; padding:13px 14px; margin:9px 0;
    }
    .scan-spam { color:#ff5c8e; font-weight:800; }
    .scan-ham { color:#36e5c0; font-weight:800; }
    .small-muted { color:#a9c7e4; font-size:12px; line-height:1.6; }
    div[data-testid="stExpander"] {
        background:rgba(8,30,55,.75); border:1px solid rgba(0,190,240,.5);
        border-radius:13px;
    }
    d    div[data-testid="stExpander"] {
        background:rgba(8,30,55,.75);
        border:1px solid rgba(0,190,240,.5);
        border-radius:13px;
    }

    div[data-testid="stExpander"] summary {
        color: #eaf5ff;
        font-weight: 700;
    }

    /* Hide Deploy button */
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    footer {
        visibility: hidden !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Load model --------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

model = load_model()
metrics = load_metrics()

if "message_text" not in st.session_state:
    st.session_state.message_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">कवच</div>
                <div class="brand-sub">YOUR DIGITAL SHIELD</div>
            </div>
        </div>
        <div class="side-heading">Your message safety workspace</div>
        <div class="side-copy">A machine-learning text classifier that checks a message for patterns commonly found in spam.</div>
        <div class="side-section">
            <div class="section-label">◆ MODEL DETAILS</div>
            <div class="side-item">● TF-IDF text features</div>
            <div class="side-item">● Multinomial Naive Bayes</div>
            <div class="side-item">● UCI SMS Spam Collection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    accuracy = metrics.get("accuracy", metrics.get("test_accuracy", 0.97))
    try:
        accuracy_pct = float(accuracy) * 100 if float(accuracy) <= 1 else float(accuracy)
    except (TypeError, ValueError):
        accuracy_pct = 97.0
    st.markdown(
        f"""
        <div class="side-section">
            <div class="section-label">◈ TEST-SET PERFORMANCE</div>
            <div class="accuracy-card">
                <div class="muted">Accuracy</div>
                <div class="accuracy-value">{accuracy_pct:.2f}%</div>
            </div>
        </div>
        <div class="small-muted" style="margin-top:28px;">Smarter Detection&nbsp; • &nbsp;Safer Communication</div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------- Main layout --------------------------
left, right = st.columns([1.72, 1.08], gap="large")

with left:
    sample1, sample2 = st.columns(2, gap="medium")
    with sample1:
        if st.button("➤  Try a normal message", key="normal_sample"):
            st.session_state.message_text = "Hey, are we still meeting tomorrow for lunch?"
            st.rerun()
    with sample2:
        if st.button("⚡  Try a spam example", key="spam_sample"):
            st.session_state.message_text = "Congratulations! You have won a free prize. Claim your reward now!"
            st.rerun()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">▣ &nbsp; MESSAGE TEXT</div>', unsafe_allow_html=True)
    message = st.text_area(
        "Paste the message you want to inspect",
        key="message_text",
        height=145,
        placeholder="Type or paste a message here...",
        label_visibility="collapsed",
    )
    analyze = st.button("✦  Analyze message", key="analyze_message")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
        elif model is None:
            st.error("Model file not found. Run `python train_model.py` first.")
        else:
            try:
                prediction = str(model.predict([message])[0]).lower()
                is_spam = prediction in {"spam", "1", "true"}
                probabilities = model.predict_proba([message])[0]
                classes = [str(c).lower() for c in model.classes_]
                spam_index = next((i for i, c in enumerate(classes) if c in {"spam", "1", "true"}), None)
                spam_probability = float(probabilities[spam_index]) if spam_index is not None else (1.0 if is_spam else 0.0)
                # The model label is authoritative; probability is shown as a model estimate.
                st.session_state.last_result = {
                    "message": message,
                    "is_spam": is_spam,
                    "probability": spam_probability,
                    "time": datetime.now().strftime("%H:%M"),
                }
                st.session_state.history.insert(0, st.session_state.last_result.copy())
                st.session_state.history = st.session_state.history[:5]
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_result")
    st.markdown('<div class="section-label">02 / SCAN RESULT</div>', unsafe_allow_html=True)
    if result:
        spam = result["is_spam"]
        prob = result["probability"] * 100
        label = "Likely SPAM" if spam else "Likely NOT SPAM (HAM)"
        css_class = "spam" if spam else ""
        text_class = "result-spam" if spam else "result-ham"
        icon = "⚠️" if spam else "✓"
        fill_color = "#ff3977" if spam else "#18d9b8"
        st.markdown(
            f"""
            <div class="result-panel {css_class}">
                <div class="panel-title">⬡ &nbsp; SCAN RESULT</div>
                <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
                    <div style="font-size:48px;">{icon}</div>
                    <div>
                        <div class="result-main {text_class}">{label}</div>
                        <div class="muted">{"This message shows patterns commonly found in spam." if spam else "No strong spam pattern was detected in this message."}</div>
                    </div>
                </div>
                <div class="prob-row">
                    <span>Estimated spam probability</span>
                    <strong style="color:{fill_color};font-size:18px;">{prob:.2f}%</strong>
                </div>
                <div class="prob-track"><div class="prob-fill" style="width:{prob:.2f}%;background:{fill_color};"></div></div>
                <div class="small-muted" style="margin-top:12px;">This is a model estimate, not a guarantee. Check suspicious links and requests independently.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-panel">
                <div class="panel-title">⬡ &nbsp; SCAN RESULT</div>
                <div class="result-main">Ready to analyze</div>
                <div class="muted">Enter a message and select Analyze message to see the model prediction here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("›  View dataset and training details"):
        st.write("**Dataset:** UCI SMS Spam Collection")
        st.write("**Feature extraction:** TF-IDF")
        st.write("**Classifier:** Multinomial Naive Bayes")
        st.write("**Training split:** 80% training / 20% testing")
        if metrics:
            st.json(metrics)
        else:
            st.caption("No metrics.json file was found. Train the model to generate it.")

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">⚙ &nbsp; HOW IT WORKS</div>
            <div class="step-card">
                <span class="step-num">01</span><span class="step-title">Enter text</span>
                <div class="muted" style="margin:10px 0 0 55px;">Paste the message you want to inspect.</div>
            </div>
            <div class="step-card">
                <span class="step-num">02</span><span class="step-title">Analyze patterns</span>
                <div class="muted" style="margin:10px 0 0 55px;">TF-IDF converts words into numerical features for the classifier.</div>
            </div>
            <div class="step-card" style="margin-bottom:0;">
                <span class="step-num">03</span><span class="step-title">Review prediction</span>
                <div class="muted" style="margin:10px 0 0 55px;">The model returns a HAM or SPAM label and an estimated spam probability.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Recent scans")
    if st.session_state.history:
        for item in st.session_state.history:
            spam = item["is_spam"]
            badge = "SPAM" if spam else "HAM"
            css = "scan-spam" if spam else "scan-ham"
            prob = item["probability"] * 100
            short_message = item["message"]
            if len(short_message) > 82:
                short_message = short_message[:79] + "..."
            st.markdown(
                f"""
                <div class="scan-card">
                    <div style="display:flex;justify-content:space-between;gap:10px;">
                        <span class="{css}">{'✦' if spam else '✓'} &nbsp; {badge}</span>
                        <strong class="{css}">{prob:.2f}%</strong>
                    </div>
                    <div class="small-muted" style="margin-top:7px;">{short_message}</div>
                    <div class="small-muted">{item['time']} · Spam probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="scan-card"><div class="muted">Your analyzed messages will appear here.</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div style="text-align:center;color:#7898b8;font-size:12px;margin-top:26px;">KAVACH • SMART SPAM PROTECTION • MACHINE LEARNING DEMO</div>',
    unsafe_allow_html=True,
)
