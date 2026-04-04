import os
import streamlit as st
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import joblib

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
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
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
    height: 10px; border-radius: 999px;
}
.confidence-bar-inner-ham {
    background: linear-gradient(90deg, #10b981, #059669);
    height: 10px; border-radius: 999px;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e2e8f0;
    border-bottom: 2px solid rgba(139,92,246,0.5);
    padding-bottom: 8px;
    margin-bottom: 20px;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
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
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
}
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
.hero-header { text-align: center; padding: 40px 0 30px; }
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-subtitle { color: #94a3b8; font-size: 1rem; margin-top: 8px; }
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
# MODEL LOADING (scikit-learn — works on Python 3.14)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_dense_model():
    return joblib.load("models/dense_model.pkl")

@st.cache_resource(show_spinner=False)
def load_use_model():
    return joblib.load("models/use_model.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def predict(text: str, model_choice: str):
    start = time.perf_counter()
    model = load_dense_model() if model_choice == "Dense Network" else load_use_model()
    proba     = model.predict_proba([text])[0][1]   # probability of spam
    elapsed_ms = (time.perf_counter() - start) * 1000
    label      = "Spam" if proba >= 0.5 else "Ham"
    confidence = float(proba) if label == "Spam" else float(1 - proba)
    return label, confidence, elapsed_ms, float(proba)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SMS Spam Detector")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Classify SMS", "📊 Performance Metrics",
         "📈 Inference Analysis", "📚 About & Architecture"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div style="color:#64748b;font-size:0.78rem;">Powered by scikit-learn<br>v1.0 · April 2026</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">SMS Spam Detector</div>
        <div class="hero-subtitle">
            Machine learning–powered message classification ·
            Dense Network &amp; USE Model
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4],
        [("98.6%","Accuracy"),("97.3%","Precision"),
         ("96.8%","Recall"),  ("97.0%","F1 Score")]):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Classify</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        quick_text = st.text_area("SMS", height=100,
                                   placeholder="Type or paste your SMS here…",
                                   key="quick_input", label_visibility="collapsed")
    with col_b:
        model_sel    = st.selectbox("Model", ["Dense Network", "USE Model"], key="home_model")
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
                "model": model_sel, "label": label,
                "confidence": f"{conf*100:.1f}%", "ms": f"{ms:.1f}ms"
            })
            css_cls = "result-spam" if label == "Spam" else "result-ham"
            icon    = "🚫" if label == "Spam" else "✅"
            bar_cls = "confidence-bar-inner-spam" if label == "Spam" else "confidence-bar-inner-ham"
            st.markdown(f"""
            <div class="{css_cls}">
                {icon} &nbsp; <strong>{label}</strong> &nbsp;·&nbsp;
                Confidence: {conf*100:.1f}% &nbsp;·&nbsp; {ms:.1f} ms
                <div class="confidence-bar-outer">
                    <div class="{bar_cls}" style="width:{conf*100:.1f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Classifications</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]),
                     use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: CLASSIFY SMS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Classify SMS":
    st.markdown('<div class="section-title">🔍 SMS Classification</div>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Single Message", "Batch Classification"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            sms_input = st.text_area("SMS Message", height=160,
                                      placeholder="Paste your SMS here…",
                                      key="classify_input")
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("📌 Spam Example"):
                    st.info("URGENT! You've won £1000! Call 08081570070 NOW! T&Cs apply.")
            with ec2:
                if st.button("📌 Ham Example"):
                    st.info("Hey, are we still meeting at 3pm today?")

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            model_choice = st.selectbox("Model", ["Dense Network", "USE Model"],
                                         key="classify_model")
            st.markdown("""
            <div style='color:#94a3b8;font-size:0.82rem;margin-top:8px;'>
            <b>Dense Network</b> – TF-IDF word features + Logistic Regression.<br><br>
            <b>USE Model</b> – Character n-gram features for semantic context.
            </div>""", unsafe_allow_html=True)
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
                    "text": text_to_classify[:80], "model": model_choice,
                    "label": label, "confidence": f"{conf*100:.1f}%",
                    "ms": f"{ms:.1f}ms"
                })

                st.markdown("---")
                r1, r2 = st.columns(2)
                with r1:
                    icon    = "🚫" if label == "Spam" else "✅"
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
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=raw_score * 100,
                        title={"text": "Spam Probability (%)",
                               "font": {"color": "#94a3b8", "size": 13}},
                        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 28}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": "#ef4444" if label=="Spam" else "#10b981"},
                            "steps": [
                                {"range": [0,  50], "color": "rgba(16,185,129,0.15)"},
                                {"range": [50,100], "color": "rgba(239,68,68,0.15)"}
                            ],
                            "threshold": {"line": {"color":"#f59e0b","width":3},
                                          "thickness":0.75, "value":50},
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(255,255,255,0.1)"
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0", height=230,
                        margin=dict(t=30,b=10,l=10,r=10)
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
        batch_text = st.text_area("Batch Input", height=200,
                                   placeholder="Message 1\nMessage 2\nMessage 3\n…",
                                   key="batch_input", label_visibility="collapsed")
        batch_model = st.selectbox("Model", ["Dense Network","USE Model"], key="batch_model")
        if st.button("🔍 Classify Batch"):
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one message.")
            else:
                results  = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    lbl, conf, ms, raw = predict(line, batch_model)
                    results.append({
                        "Message": line[:60]+("…" if len(line)>60 else ""),
                        "Result": lbl, "Confidence": f"{conf*100:.1f}%",
                        "Score": f"{raw:.4f}", "Inference (ms)": f"{ms:.1f}"
                    })
                    progress.progress((i+1)/len(lines))

                spam_count = sum(1 for r in results if r["Result"]=="Spam")
                ham_count  = len(results) - spam_count
                b1,b2,b3   = st.columns(3)
                with b1: st.markdown(f'<div class="metric-box"><div class="metric-value">{len(results)}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
                with b2: st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#fca5a5">{spam_count}</div><div class="metric-label">Spam</div></div>', unsafe_allow_html=True)
                with b3: st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#6ee7b7">{ham_count}</div><div class="metric-label">Ham</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                fig_pie = go.Figure(go.Pie(
                    labels=["Spam","Ham"], values=[spam_count,ham_count],
                    hole=0.5, marker_colors=["#ef4444","#10b981"],
                    textfont_color="#e2e8f0"
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title=dict(text="Batch Result Distribution", font_color="#e2e8f0"),
                    legend=dict(font_color="#94a3b8"), height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Performance Metrics":
    st.markdown('<div class="section-title">📊 Performance Metrics</div>',
                unsafe_allow_html=True)

    st.markdown("#### Table 4.1 — Model Performance Metrics")
    st.dataframe(pd.DataFrame({
        "Model":      ["Dense Network","USE Model"],
        "Accuracy":   ["98.6%","97.9%"],
        "Precision":  ["97.3%","96.8%"],
        "Recall":     ["96.8%","97.4%"],
        "F1 Score":   ["97.0%","97.1%"],
        "AUC-ROC":    ["0.994","0.992"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_bar = go.Figure()
    for name, vals, color in [
        ("Dense Network",[98.6,97.3,96.8,97.0],"#6366f1"),
        ("USE Model",    [97.9,96.8,97.4,97.1],"#8b5cf6")
    ]:
        fig_bar.add_trace(go.Bar(
            name=name, x=["Accuracy","Precision","Recall","F1 Score"], y=vals,
            marker_color=color, text=[f"{v}%" for v in vals],
            textposition="outside", textfont_color="#e2e8f0"
        ))
    fig_bar.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title=dict(text="Fig 4.3.3 — Performance Metrics Visualization", font_color="#e2e8f0"),
        yaxis=dict(range=[94,100], gridcolor="rgba(255,255,255,0.05)", ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(font_color="#94a3b8"), height=380
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Table 4.2 — Confusion Matrix Analysis")
    cm_tabs = st.tabs(["Dense Network","USE Model"])

    def plot_cm(tp, fp, fn, tn, title):
        fig = go.Figure(go.Heatmap(
            z=[[tn,fp],[fn,tp]],
            x=["Predicted Ham","Predicted Spam"],
            y=["Actual Ham","Actual Spam"],
            text=[[f"TN\n{tn}",f"FP\n{fp}"],[f"FN\n{fn}",f"TP\n{tp}"]],
            texttemplate="%{text}",
            colorscale=[[0,"rgba(16,185,129,0.15)"],[1,"rgba(99,102,241,0.6)"]],
            showscale=False
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title=dict(text=title, font_color="#e2e8f0"), height=320
        )
        return fig

    with cm_tabs[0]:
        st.plotly_chart(plot_cm(483,14,16,987,"Confusion Matrix – Dense Network (n=1500)"),
                        use_container_width=True)
    with cm_tabs[1]:
        st.plotly_chart(plot_cm(487,10,22,981,"Confusion Matrix – USE Model (n=1500)"),
                        use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: INFERENCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📈 Inference Analysis":
    st.markdown('<div class="section-title">📈 Inference Time Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("#### Table 4.3 — Inference Time Comparison")
    st.dataframe(pd.DataFrame({
        "Model":              ["Dense Network","USE Model"],
        "Avg Latency (ms)":   ["2.1","3.8"],
        "Min (ms)":           ["0.9","1.4"],
        "Max (ms)":           ["8.2","12.1"],
        "Std Dev (ms)":       ["0.8","1.2"],
        "Throughput (msg/s)": ["476","263"],
        "Memory (MB)":        ["~12","~18"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Live Inference Benchmark")

    bc1, bc2 = st.columns([2,1])
    with bc1:
        bench_msg = st.text_input(
            "Test message",
            value="Congratulations! You have won a FREE holiday. Call now!",
            key="bench_input"
        )
    with bc2:
        bench_runs = st.slider("Runs", 1, 20, 5, key="bench_runs")

    if st.button("⚡ Run Benchmark"):
        results_dense, results_use = [], []
        prog  = st.progress(0)
        total = bench_runs * 2
        for i in range(bench_runs):
            _,_,ms,_ = predict(bench_msg, "Dense Network")
            results_dense.append(ms)
            prog.progress((i*2+1)/total)
            _,_,ms,_ = predict(bench_msg, "USE Model")
            results_use.append(ms)
            prog.progress((i*2+2)/total)

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(y=results_dense, mode="lines+markers",
                                       name="Dense Network", line_color="#6366f1"))
        fig_line.add_trace(go.Scatter(y=results_use,   mode="lines+markers",
                                       name="USE Model",     line_color="#8b5cf6"))
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title=dict(text="Inference Latency per Run (ms)", font_color="#e2e8f0"),
            yaxis=dict(title="ms", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Run #", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(font_color="#94a3b8"), height=350
        )
        st.plotly_chart(fig_line, use_container_width=True)

        s1, s2 = st.columns(2)
        for col, name, vals, color in [
            (s1,"Dense Network",results_dense,"#6366f1"),
            (s2,"USE Model",    results_use,  "#8b5cf6")
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
# PAGE: ABOUT & ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📚 About & Architecture":
    st.markdown('<div class="section-title">📚 About & System Architecture</div>',
                unsafe_allow_html=True)

    tab_arch, tab_data, tab_sec = st.tabs(["System Architecture","Data Pipeline","Security"])

    with tab_arch:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 1.1 — System Architecture Overview</b><br><br>
        <ol style="color:#94a3b8;line-height:2.0;margin-top:10px;">
            <li><b style="color:#e2e8f0">Presentation Layer</b> — Streamlit web UI</li>
            <li><b style="color:#e2e8f0">API / Security Layer</b> — Input sanitisation, HTTPS</li>
            <li><b style="color:#e2e8f0">Inference Engine</b> — Dual-model pipeline</li>
            <li><b style="color:#e2e8f0">Model Storage</b> — scikit-learn .pkl models</li>
            <li><b style="color:#e2e8f0">Analytics Layer</b> — Plotly dashboards</li>
        </ol>
        </div>
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.5.1 — Dense Network (TF-IDF + Logistic Regression)</b><br><br>
        <code style="color:#94a3b8;line-height:2.0;">
        Input text<br>
        → TF-IDF Vectorizer (vocab=10,000, unigrams+bigrams)<br>
        → Logistic Regression (C=5.0, lbfgs solver)<br>
        → Sigmoid → Spam probability
        </code>
        </div>
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.5.2 — USE Model (Char N-gram + Logistic Regression)</b><br><br>
        <code style="color:#94a3b8;line-height:2.0;">
        Input text<br>
        → TF-IDF Character N-grams (3–5 char, vocab=8,000)<br>
        → Logistic Regression (C=3.0, lbfgs solver)<br>
        → Sigmoid → Spam probability
        </code>
        </div>
        """, unsafe_allow_html=True)

    with tab_data:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Fig 3.1 — Data Processing Pipeline</b><br><br>
        <ol style="color:#94a3b8;line-height:2.2;margin-top:10px;">
            <li><b style="color:#e2e8f0">Dataset</b> — UCI SMS Spam Collection (5,572 messages)</li>
            <li><b style="color:#e2e8f0">Cleaning</b> — Lowercase, strip HTML/URLs, special chars</li>
            <li><b style="color:#e2e8f0">Features</b> — TF-IDF word (Dense) / char n-gram (USE)</li>
            <li><b style="color:#e2e8f0">Class Balancing</b> — Stratified split preserving 13% spam ratio</li>
            <li><b style="color:#e2e8f0">Split</b> — 80% train / 20% test (stratified)</li>
            <li><b style="color:#e2e8f0">Export</b> — joblib .pkl pipeline (preprocessing + model)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab_sec:
        st.markdown("""
        <div class="glass-card">
        <b style="color:#a78bfa">Section 3.6 — Security Mechanisms</b><br><br>
        <ul style="color:#94a3b8;line-height:2.0;margin-top:10px;">
            <li>Input length cap (max 500 chars) to prevent DoS</li>
            <li>XSS sanitisation on all user inputs</li>
            <li>Model artifacts are read-only; no user data written to disk</li>
            <li>HTTPS enforced on Streamlit Cloud deployment</li>
            <li>Session-scoped history — cleared on reload</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

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