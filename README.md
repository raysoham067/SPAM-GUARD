# 🛡️ SMS Spam Detection System

A deep-learning web application that classifies SMS messages as **Spam** or **Ham** using two TensorFlow models — a lightweight **Dense Network** and a semantically rich **Universal Sentence Encoder (USE)** model.

---

## 📋 Table of Contents

| Chapter | Topic |
|---------|-------|
| 1 | Introduction & Project Overview |
| 2 | Literature Review (ML/NLP/DL) |
| 3 | Methodology & Architecture |
| 4 | Results & Performance Metrics |
| 5 | Future Scope |
| 6 | Conclusion |

---

## 🚀 Quick Start

### 1. Clone / Download the project
```bash
# If using git:
git clone <repo-url>
cd sms-spam-project

# Or just unzip the folder you already have
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
sms-spam-project/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
└── models/
    ├── spam_dense.keras      ← Dense Network model
    └── spam_use_model.keras  ← USE-based model
```

---

## 🖥️ Application Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview dashboard with quick classify widget & history |
| 🔍 Classify SMS | Single message + batch classification with confidence gauge |
| 📊 Performance Metrics | Table 4.1 (metrics), Table 4.2 (confusion matrix), Fig 4.3.3 |
| 📈 Inference Analysis | Table 4.3 latency stats + live benchmark tool |
| 📚 About & Architecture | System diagrams, data pipeline, security, future scope |

---

## 🧠 Models

### Dense Network (Fig 3.5.1)
```
Input text
  → TextVectorization (vocab=10,000, seq_len=128)
  → Embedding (dim=64)
  → GlobalAveragePooling1D
  → Dense(128, relu) → Dropout(0.3)
  → Dense(64, relu)  → Dropout(0.2)
  → Dense(1, sigmoid)
```
- **Avg latency**: ~12 ms  |  **Accuracy**: 98.6%

### USE Model (Fig 3.5.2)
```
Input text
  → Universal Sentence Encoder (512-dim)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu)
  → Dense(1, sigmoid)
```
- **Avg latency**: ~90 ms  |  **Accuracy**: 97.9%

---

## 📊 Performance Summary (Table 4.1)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Dense Network | 98.6% | 97.3% | 96.8% | 97.0% |
| USE Model     | 97.9% | 96.8% | 97.4% | 97.1% |

---

## ⏱️ Inference Time (Table 4.3)

| Model | Avg (ms) | Min (ms) | Max (ms) | Throughput |
|-------|----------|----------|----------|------------|
| Dense Network | 12.4 | 8.1 | 31.2 | 80.6 msg/s |
| USE Model     | 89.7 | 71.3 | 142.6 | 11.1 msg/s |

---

## 🔒 Security Features (Section 3.6)
- Input length capped at 500 characters
- XSS sanitisation on all user inputs
- Read-only model artifacts — no user data written to disk
- Session-scoped history (cleared on reload)

---

## 🔭 Future Scope (Chapter 5)
1. **Multi-Language Support** — Hindi, Bengali, Spanish SMS detection
2. **Real-Time API** — FastAPI/Flask REST endpoint for mobile integration
3. **Advanced Pattern Recognition** — Regex + embedding hybrid pipeline
4. **User Feedback & Retraining** — Active learning loop
5. **Phishing Detection** — URL and domain reputation scoring
6. **Mobile App** — React Native / Flutter front-end

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend  | Streamlit, Plotly |
| ML Framework | TensorFlow / Keras |
| Embeddings | Universal Sentence Encoder (TF-Hub) |
| Language | Python 3.10+ |
| Visualisation | Plotly Express / Graph Objects |

---

## 📄 License
Academic project — for educational use only.
