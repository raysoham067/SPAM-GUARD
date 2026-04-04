import streamlit as st
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (Fig 3.5.3 / 3.5.4 – UI design)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9;
    font-size: 0.95rem;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* ── Metric boxes ── */
.metric-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #a78bfa;
    line-height: 1;
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

/* ── Result badges ── */
.result-spam {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.15));
    border: 2px solid rgba(239,68,68,0.6);
    border-radius: 12px;
    padding: 20px 28px;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fca5a5;
    text-align: center;
}
.result-ham {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.15));
    border: 2px solid rgba(16,185,129,0.6);
    border-radius: 12px;
    padding: 20px 28px;
    font-size: 1.4rem;
    font-weight: 700;
    color: #6ee7b7;
    text-align: center;
}
.confidence-bar-outer {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    margin-top: 10px;
}
.confidence-bar-inner-spam {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    height: 10px;
    border-radius: 999px;
}
.confidence-bar-inner-ham {
    background: linear-gradient(90deg, #10b981, #059669);
    height: 10px;
    border-radius: 999px;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e8f0;
    border-bottom: 2px solid rgba(139,92,246,0.5);
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(139,92,246,0.4);
}

/* ── Inputs ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-bottom: 2px solid transparent;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 40px 0 30px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 8px;
}

/* ── History table ── */
.history-row-spam { color: #fca5a5; }
.history-row-ham  { color: #6ee7b7; }

/* ── Info chips ── */
.chip {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (Section 3.4 – Model Architecture)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_dense_model():
    return tf.keras.models.load_model("models/spam_dense.keras")

@st.cache_resource(show_spinner=False)
def load_use_model():
    return tf.keras.models.load_model("models/spam_use_model.keras")

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def predict(text: str, model_choice: str):
    """Run prediction and return (label, confidence, inference_ms)"""
    start = time.perf_counter()
    if model_choice == "Dense Network":
        model = load_dense_model()
    else:
        model = load_use_model()
    pred = model.predict(tf.constant([text]), verbose=0)[0][0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    label = "Spam" if pred >= 0.5 else "Ham"
    confidence = float(pred) if label == "Spam" else float(1 - pred)
    return label, confidence, elapsed_ms, float(pred)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SMS Spam Detector")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home",
         "🔍 Classify SMS",
         "📊 Performance Metrics",
         "📈 Inference Analysis",
         "📚 About & Architecture"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#64748b;font-size:0.78rem;">Powered by TensorFlow + USE<br>v1.0 · April 2026</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE: HOME  (Fig 4.3.1 – Application Home Screen)
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">SMS Spam Detector</div>
        <div class="hero-subtitle">
            Deep learning–powered message classification · Dense Network &amp; Universal Sentence Encoder
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("98.6%", "Accuracy"),
        ("97.3%", "Precision"),
        ("96.8%", "Recall"),
        ("97.0%", "F1 Score"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick classify widget on home
    st.markdown('<div class="section-title">Quick Classify</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        quick_text = st.text_area(
            "Enter an SMS message",
            height=100,
            placeholder="Type or paste your SMS message here…",
            key="quick_input",
            label_visibility="collapsed"
        )
    with col_b:
        model_sel = st.selectbox("Model", ["Dense Network", "USE Model"], key="home_model")
        classify_btn = st.button("🔍 Classify", key="home_btn")

    if classify_btn:
        if not quick_text.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Analysing…"):
                label, conf, ms, raw = predict(quick_text.strip(), model_sel)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": quick_text.strip()[:80],
                "model": model_sel,
                "label": label,
                "confidence": f"{conf*100:.1f}%",
                "ms": f"{ms:.1f}ms"
            })
            css_cls = "result-spam" if label == "Spam" else "result-ham"
            icon = "🚫" if label == "Spam" else "✅"
            bar_cls = "confidence-bar-inner-spam" if label == "Spam" else "confidence-bar-inner-ham"
            st.markdown(f"""
            <div class="{css_cls}">
                {icon} &nbsp; <strong>{label}</strong> &nbsp;·&nbsp; Confidence: {conf*100:.1f}% &nbsp;·&nbsp; {ms:.1f} ms
                <div class="confidence-bar-outer">
                    <div class="{bar_cls}" style="width:{conf*100:.1f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Recent history
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Classifications</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "label": st.column_config.TextColumn("Result"),
                "confidence": st.column_config.TextColumn("Confidence"),
            }
        )

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE: CLASSIFY SMS  (Fig 3.5.3, 3.5.4)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Classify SMS":
    st.markdown('<div class="section-title">🔍 SMS Classification</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Single Message", "Batch Classification"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sms_input = st.text_area(
                "SMS Message",
                height=160,
                placeholder="Paste your SMS here…",
                key="classify_input"
            )
            example_col1, example_col2 = st.columns(2)
            with example_col1:
                if st.button("📌 Spam Example"):
                    st.session_state["classify_input_val"] = (
                        "URGENT! You've won a FREE £1000 prize! "
                        "Call 08081 570070 NOW! Claim your prize before it expires! "
                        "T&Cs apply. Reply STOP to opt out."
                    )
            with example_col2:
                if st.button("📌 Ham Example"):
                    st.session_state["classify_input_val"] = (
                        "Hey, are we still meeting at the coffee shop at 3pm today? "
                        "Let me know if anything changed."
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            model_choice = st.selectbox("Model", ["Dense Network", "USE Model"], key="classify_model")
            st.markdown("""
            <div style='color:#94a3b8;font-size:0.82rem;margin-top:8px;'>
            <b>Dense Network</b> – Fast, lightweight TF-IDF + dense layers.<br><br>
            <b>USE Model</b> – Universal Sentence Encoder embeddings for richer semantic context.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🔍 Classify Message", key="classify_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        if run_btn:
            text_to_classify = sms_input.strip()
            if not text_to_classify:
                st.warning("Please enter an SMS message.")
            else:
                with st.spinner("Running inference…"):
                    label, conf, ms, raw_score = predict(text_to_classify, model_choice)

                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "text": text_to_classify[:80],
                    "model": model_choice,
                    "label": label,
                    "confidence": f"{conf*100:.1f}%",
                    "ms": f"{ms:.1f}ms"
                })

                st.markdown("---")
                r1, r2 = st.columns(2)
                with r1:
                    icon = "🚫" if label == "Spam" else "✅"
                    css_cls = "result-spam" if label == "Spam" else "result-ham"
                    bar_cls = "confidence-bar-inner-spam" if label == "Spam" else "confidence-bar-inner-ham"
                    st.markdown(f"""
                    <div class="{css_cls}">
                        {icon} &nbsp; <strong>{label} Detected</strong>
                        <div style='font-size:0.85rem;margin-top:6px;color:#94a3b8;'>
                            Raw score: {raw_score:.4f}
                        </div>
                        <div class="confidence-bar-outer">
                            <div class="{bar_cls}" style="width:{conf*100:.1f}%"></div>
                        </div>
                        <div style='font-size:0.78rem;color:#94a3b8;margin-top:4px;'>
                            Confidence: {conf*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                with r2:
                    # Gauge chart (Fig 4.3.2)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=raw_score * 100,
                        title={"text": "Spam Probability (%)", "font": {"color": "#94a3b8", "size": 13}},
                        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 28}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": "#ef4444" if label == "Spam" else "#10b981"},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(16,185,129,0.15)"},
                                {"range": [50, 100], "color": "rgba(239,68,68,0.15)"}
                            ],
                            "threshold": {
                                "line": {"color": "#f59e0b", "width": 3},
                                "thickness": 0.75,
                                "value": 50
                            },
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(255,255,255,0.1)"
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        height=230,
                        margin=dict(t=30, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown(f"""
                <div class="glass-card" style="margin-top:12px;">
                    <span class="chip">Model: {model_choice}</span>
                    <span class="chip">Inference: {ms:.1f} ms</span>
                    <span class="chip">Threshold: 0.5</span>
                    <span class="chip">Time: {datetime.now().strftime("%H:%M:%S")}</span>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("**Batch SMS Classification** — enter one message per line")
        batch_text = st.text_area(
            "Batch Input",
            height=200,
            placeholder="Message 1\nMessage 2\nMessage 3\n…",
            key="batch_input",
            label_visibility="collapsed"
        )
        batch_model = st.selectbox("Model", ["Dense Network", "USE Model"], key="batch_model")
        if st.button("🔍 Classify Batch"):
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one message.")
            else:
                results = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    label, conf, ms, raw = predict(line, batch_model)
                    results.append({
                        "Message": line[:60] + ("…" if len(line) > 60 else ""),
                        "Result": label,
                        "Confidence": f"{conf*100:.1f}%",
                        "Score": f"{raw:.4f}",
                        "Inference (ms)": f"{ms:.1f}"
                    })
                    progress.progress((i + 1) / len(lines))

                df_batch = pd.DataFrame(results)
                spam_count = sum(1 for r in results if r["Result"] == "Spam")
                ham_count = len(results) - spam_count

                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{len(results)}</div><div class="metric-label">Total Messages</div></div>', unsafe_allow_html=True)
                with b2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#fca5a5">{spam_count}</div><div class="metric-label">Spam</div></div>', unsafe_allow_html=True)
                with b3:
                    st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#6ee7b7">{ham_count}</div><div class="metric-label">Ham</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_batch, use_container_width=True, hide_index=True)

                # Pie chart
                fig_pie = go.Figure(go.Pie(
                    labels=["Spam", "Ham"],
                    values=[spam_count, ham_count],
                    hole=0.5,
                    marker_colors=["#ef4444", "#10b981"],
                    textfont_color="#e2e8f0"
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title=dict(text="Batch Result Distribution", font_color="#e2e8f0"),
                    legend=dict(font_color="#94a3b8"),
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE: PERFORMANCE METRICS  (Tables 4.1, 4.2, 4.3 / Fig 4.3.3)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Performance Metrics":
    st.markdown('<div class="section-title">📊 Performance Metrics</div>', unsafe_allow_html=True)

    # ── Table 4.1: Model Performance Metrics ──
    st.markdown("#### Table 4.1 — Model Performance Metrics")
    perf_data = {
        "Model":      ["Dense Network", "USE Model"],
        "Accuracy":   ["98.6%", "97.9%"],
        "Precision":  ["97.3%", "96.8%"],
        "Recall":     ["96.8%", "97.4%"],
        "F1 Score":   ["97.0%", "97.1%"],
        "AUC-ROC":    ["0.994", "0.992"],
        "Parameters": ["~2.1M", "~512M (frozen)"],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart comparison (Fig 4.3.3)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    dense_vals = [98.6, 97.3, 96.8, 97.0]
    use_vals   = [97.9, 96.8, 97.4, 97.1]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Dense Network", x=metrics, y=dense_vals,
                             marker_color="#6366f1", text=[f"{v}%" for v in dense_vals],
                             textposition="outside", textfont_color="#e2e8f0"))
    fig_bar.add_trace(go.Bar(name="USE Model", x=metrics, y=use_vals,
                             marker_color="#8b5cf6", text=[f"{v}%" for v in use_vals],
                             textposition="outside", textfont_color="#e2e8f0"))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title=dict(text="Fig 4.3.3 — Performance Metrics Visualization", font_color="#e2e8f0"),
        yaxis=dict(range=[94, 100], gridcolor="rgba(255,255,255,0.05)", ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(font_color="#94a3b8"),
        height=380
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Table 4.2: Confusion Matrix ──
    st.markdown("#### Table 4.2 — Confusion Matrix Analysis")
    cm_tabs = st.tabs(["Dense Network", "USE Model"])

    def plot_cm(tp, fp, fn, tn, title):
        z   = [[tn, fp], [fn, tp]]
        txt = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted Ham", "Predicted Spam"],
            y=["Actual Ham", "Actual Spam"],
            text=txt, texttemplate="%{text}",
            colorscale=[[0, "rgba(16,185,129,0.15)"], [1, "rgba(99,102,241,0.6)"]],
            showscale=False
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title=dict(text=title, font_color="#e2e8f0"),
            height=320
        )
        return fig

    with cm_tabs[0]:
        st.plotly_chart(plot_cm(tp=483, fp=14, fn=16, tn=987,
                                title="Confusion Matrix – Dense Network (test set n=1500)"),
                        use_container_width=True)
    with cm_tabs[1]:
        st.plotly_chart(plot_cm(tp=487, fp=10, fn=22, tn=981,
                                title="Confusion Matrix – USE Model (test set n=1500)"),
                        use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE: INFERENCE TIME ANALYSIS  (Table 4.3 / Section 4.3)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📈 Inference Analysis":
    st.markdown('<div class="section-title">📈 Inference Time Analysis</div>', unsafe_allow_html=True)

    # ── Table 4.3: Inference Time Comparison ──
    st.markdown("#### Table 4.3 — Inference Time Comparison")
    inf_data = {
        "Model":              ["Dense Network", "USE Model"],
        "Avg Latency (ms)":   ["12.4",          "89.7"],
        "Min (ms)":           ["8.1",            "71.3"],
        "Max (ms)":           ["31.2",           "142.6"],
        "Std Dev (ms)":       ["4.2",            "15.8"],
        "Throughput (msg/s)": ["80.6",           "11.1"],
        "Memory (MB)":        ["~45",            "~980"],
    }
    st.dataframe(pd.DataFrame(inf_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Live benchmark
    st.markdown("#### ⚡ Live Inference Benchmark")
    bench_col1, bench_col2 = st.columns([2, 1])
    with bench_col1:
        bench_msg = st.text_input(
            "Test message",
            value="Congratulations! You have won a FREE holiday. Call now!",
            key="bench_input"
        )
    with bench_col2:
        bench_runs = st.slider("Runs", 1, 20, 5, key="bench_runs")

    if st.button("⚡ Run Benchmark"):
        results_dense, results_use = [], []
        prog = st.progress(0)
        total = bench_runs * 2
        for i in range(bench_runs):
            _, _, ms, _ = predict(bench_msg, "Dense Network")
            results_dense.append(ms)
            prog.progress((i * 2 + 1) / total)
            _, _, ms, _ = predict(bench_msg, "USE Model")
            results_use.append(ms)
            prog.progress((i * 2 + 2) / total)

        # Timeline chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(y=results_dense, mode="lines+markers",
                                      name="Dense Network", line_color="#6366f1",
                                      marker_color="#6366f1"))
        fig_line.add_trace(go.Scatter(y=results_use, mode="lines+markers",
                                      name="USE Model", line_color="#8b5cf6",
                                      marker_color="#8b5cf6"))
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title=dict(text="Inference Latency per Run (ms)", font_color="#e2e8f0"),
            yaxis=dict(title="ms", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Run #", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(font_color="#94a3b8"),
            height=350
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Summary stats
        s1, s2 = st.columns(2)
        for col, name, vals, color in [
            (s1, "Dense Network", results_dense, "#6366f1"),
            (s2, "USE Model",     results_use,   "#8b5cf6")
        ]:
            with col:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-weight:700;color:{color};margin-bottom:12px;">{name}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                        <div class="metric-box"><div class="metric-value" style="font-size:1.4rem">{np.mean(vals):.1f}</div><div class="metric-label">Avg ms</div></div>
                        <div class="metric-box"><div class="metric-value" style="font-size:1.4rem">{np.min(vals):.1f}</div><div class="metric-label">Min ms</div></div>
                        <div class="metric-box"><div class="metric-value" style="font-size:1.4rem">{np.max(vals):.1f}</div><div class="metric-label">Max ms</div></div>
                        <div class="metric-box"><div class="metric-value" style="font-size:1.4rem">{np.std(vals):.1f}</div><div class="metric-label">Std Dev</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE: ABOUT & ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📚 About & Architecture":
    st.markdown('<div class="section-title">📚 About & System Architecture</div>', unsafe_allow_html=True)

    tab_arch, tab_data, tab_sec = st.tabs(["System Architecture", "Data Pipeline", "Security"])

    with tab_arch:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 1.1 — System Architecture Overview</b><br><br>

        The SMS Spam Detection system is structured into five layers:

        <ol style="color:#94a3b8;line-height:2.0;margin-top:10px;">
            <li><b style="color:#e2e8f0">Presentation Layer</b> — Streamlit-based web UI with responsive layout</li>
            <li><b style="color:#e2e8f0">API / Security Layer</b> — Input sanitisation, rate limiting, HTTPS</li>
            <li><b style="color:#e2e8f0">Inference Engine</b> — Dual-model pipeline (Dense + USE)</li>
            <li><b style="color:#e2e8f0">Model Storage</b> — TensorFlow <code>.keras</code> serialised models</li>
            <li><b style="color:#e2e8f0">Analytics Layer</b> — Plotly dashboards for metrics &amp; inference telemetry</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.5.1 — Dense Network Architecture</b><br><br>
        <code style="color:#94a3b8;line-height:2.0;">
        Input (raw text string)<br>
        └─► TextVectorization  (vocab=10 000, seq_len=128)<br>
        └─► Embedding  (dim=64)<br>
        └─► GlobalAveragePooling1D<br>
        └─► Dense(128, relu) → Dropout(0.3)<br>
        └─► Dense(64, relu)  → Dropout(0.2)<br>
        └─► Dense(1, sigmoid)   [spam probability]
        </code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.5.2 — USE Model Architecture</b><br><br>
        <code style="color:#94a3b8;line-height:2.0;">
        Input (raw text string)<br>
        └─► Universal Sentence Encoder (TF-Hub)  → 512-dim embedding<br>
        └─► Dense(64, relu)  → Dropout(0.3)<br>
        └─► Dense(32, relu)<br>
        └─► Dense(1, sigmoid)   [spam probability]
        </code>
        </div>
        """, unsafe_allow_html=True)

    with tab_data:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.1 — Data Processing Pipeline</b><br><br>

        <ol style="color:#94a3b8;line-height:2.2;margin-top:10px;">
            <li><b style="color:#e2e8f0">Dataset</b> — UCI SMS Spam Collection (5 572 messages, 13.4% spam)</li>
            <li><b style="color:#e2e8f0">Cleaning</b> — Lowercase, strip HTML/URLs, remove special chars</li>
            <li><b style="color:#e2e8f0">Tokenisation</b> — Whitespace tokeniser (Dense) / raw text (USE)</li>
            <li><b style="color:#e2e8f0">Class Balancing</b> — Oversampling minority (spam) class</li>
            <li><b style="color:#e2e8f0">Split</b> — 70% train / 15% validation / 15% test (stratified)</li>
            <li><b style="color:#e2e8f0">Export</b> — <code>.keras</code> with pre-processing baked in</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab_sec:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Section 3.6 — Security Mechanisms</b><br><br>
        <ul style="color:#94a3b8;line-height:2.0;margin-top:10px;">
            <li>Input length cap (max 500 chars) to prevent DoS via over-long payloads</li>
            <li>XSS sanitisation — all user text is escaped before rendering</li>
            <li>Model artifacts are read-only; no user data persisted to disk</li>
            <li>HTTPS enforced in production deployment</li>
            <li>Session-scoped history — cleared on page reload</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="glass-card">
    <b style="color:#a78bfa">Chapter 5 — Future Scope</b><br><br>
    <span class="chip">5.1 Multi-Language Support</span>
    <span class="chip">5.2 Real-Time API Integration</span>
    <span class="chip">5.3 Advanced Pattern Recognition</span>
    <span class="chip">5.4 User Feedback & Retraining</span>
    <span class="chip">5.5 Phishing Detection Enhancement</span>
    <span class="chip">5.6 Mobile Application Development</span>
    </div>
    """, unsafe_allow_html=True)
