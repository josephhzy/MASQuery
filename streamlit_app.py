"""
Streamlit frontend for MAS RegQ&A.
Calls the FastAPI backend at http://localhost:8000.
"""

import markdown as md_lib
import os
import re

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAS RegQ&A",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fb; }

    /* Force dark text on main content area */
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] li,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] * {
        color: #1a1a1a !important;
    }

    /* Force light text on sidebar (dark background) */
    section[data-testid="stSidebar"] {
        background-color: #1e1e2e !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
        color: #f0f0f0 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #aaaaaa !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #444 !important;
    }
    section[data-testid="stSidebar"] table,
    section[data-testid="stSidebar"] th,
    section[data-testid="stSidebar"] td {
        color: #f0f0f0 !important;
        border-color: #555 !important;
    }

    /* Answer card */
    .answer-card {
        background: #ffffff;
        border-left: 5px solid #d32f2f;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        font-size: 15px;
        line-height: 1.7;
        color: #1a1a1a !important;
    }
    .answer-card *  { color: #1a1a1a !important; }
    .answer-card li { color: #1a1a1a !important; margin-bottom: 6px; }
    .answer-card strong, .answer-card b { font-weight: 700 !important; color: #1a1a1a !important; }
    .answer-card em, .answer-card i     { font-style: italic !important; }
    .answer-card p  { margin-bottom: 10px; }
    .answer-card ol, .answer-card ul { padding-left: 20px; margin-bottom: 10px; }

    /* History card */
    .history-card {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .history-q { font-weight: 600; color: #333; font-size: 14px; }
    .history-meta { font-size: 11px; color: #999; margin-top: 3px; }

    /* Quality indicator bar */
    .quality-bar {
        background: #f0f4ff;
        border: 1px solid #dce4ff;
        border-radius: 8px;
        padding: 10px 18px;
        margin: 8px 0 16px 0;
        font-size: 13px;
        color: #333 !important;
    }

    /* Source card */
    .source-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    .source-doc  { font-weight: 600; color: #d32f2f; font-size: 14px; }
    .source-meta { font-size: 12px; color: #666; margin-top: 2px; }
    .source-text { font-size: 13px; color: #333; margin-top: 8px; line-height: 1.6;
                   border-left: 3px solid #eee; padding-left: 10px; }

    /* Confidence badges */
    .badge-high   { background:#e8f5e9; color:#2e7d32; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
    .badge-medium { background:#fff8e1; color:#f57f17; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
    .badge-low    { background:#fce4ec; color:#c62828; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }

    /* Status dots */
    .dot-green  { color: #4caf50; font-size: 10px; }
    .dot-red    { color: #f44336; font-size: 10px; }
    .dot-orange { color: #ff9800; font-size: 10px; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────

def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def fetch_documents() -> list:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json().get("documents", []) if r.ok else []
    except Exception:
        return []


def query_api(question: str, top_k: int) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"error": "connection_error", "detail": str(e)}


def confidence_badge(level: str) -> str:
    cls = f"badge-{level}"
    label = level.upper()
    return f'<span class="{cls}">● {label} CONFIDENCE</span>'


def status_dot(ok: bool) -> str:
    return '<span class="dot-green">●</span>' if ok else '<span class="dot-red">●</span>'


# ── Session state ──────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {question, answer, confidence, model, verified, total}


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 MAS RegQ&A")
    st.markdown("**Regulatory Q&A powered by Retrieval-Augmented Generation (RAG)**")
    st.divider()

    # About section
    st.markdown("### About This Project")
    st.markdown("""
This system lets you ask plain-English questions about **MAS (Monetary Authority of Singapore)** regulatory guidelines and get accurate, cited answers grounded in the actual documents.

**How it works:**
1. 📄 PDFs are ingested and split into chunks
2. 🔍 Each chunk is embedded into a vector using a sentence transformer model
3. 🗄️ Chunks are stored in a **FAISS** vector index
4. ❓ Your question is embedded and matched against the index
5. 🤖 The top matching chunks are sent to the configured LLM to generate an answer
6. 🔗 Sources are traced back to the exact document, section, and page number
""")
    st.divider()

    # What to try
    st.markdown("### 💡 Try Asking")
    st.markdown("""
- *What are the access control requirements under the TRM Guidelines?*
- *What must a financial institution include in its business continuity plan?*
- *What are the due diligence requirements for material outsourcing?*
- *What are MAS's requirements for incident reporting?*
- *What are the liability limits for unauthorised e-payment transactions?*
""")
    st.divider()

    # Tech stack
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
| Component | Technology |
|---|---|
| Vector Search | FAISS (dense) + BM25 (opt-in hybrid) |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | gpt-5-nano (default) · Claude Haiku (opt-in) |
| API | FastAPI |
| Frontend | Streamlit |
""")
    st.divider()

    # Health status
    st.markdown("### System Status")
    health = fetch_health()

    if health is None:
        st.error("Cannot reach API — is the server running?")
    else:
        overall = health.get("status", "unknown")
        colour = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(overall, "⚪")
        st.markdown(f"{colour} **{overall.capitalize()}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"{status_dot(health.get('faiss_index_loaded'))} Index",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"{status_dot(health.get('embedding_model_loaded'))} Embeddings",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"{status_dot(health.get('api_key_configured'))} API Key",
                unsafe_allow_html=True,
            )
            provider = health.get("llm_provider", "anthropic")
            st.markdown(f"⚙️ `{provider}`")

        st.caption(f"Index size: {health.get('faiss_index_size', 0):,} vectors")

    st.divider()

    # Documents
    st.markdown("### Indexed Documents")
    docs = fetch_documents()
    if docs:
        for doc in docs:
            st.markdown(
                f"📄 **{doc['name']}**  \n"
                f"<span style='font-size:12px;color:#888'>"
                f"{doc['chunk_count']} chunks · {doc['page_count']} pages</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed. Run POST /ingest first.")

    st.divider()

    # Settings
    st.markdown("### Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)


# ── Main area ──────────────────────────────────────────────────────────

st.markdown("# 🏦 MAS Regulatory Q&A")
st.markdown(
    "Ask any question about MAS regulatory guidelines. "
    "Answers are grounded in the indexed documents with full source tracing."
)
st.divider()

# Question input
question = st.text_area(
    "Your question",
    placeholder="e.g. What are the access control requirements under the TRM Guidelines?",
    height=100,
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    submitted = st.button("Ask →", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        "<span style='font-size:12px;color:#999;line-height:2.5'>Minimum 10 characters</span>",
        unsafe_allow_html=True,
    )

# ── Results ────────────────────────────────────────────────────────────

if submitted:
    if len(question.strip()) < 10:
        st.warning("Please enter a question of at least 10 characters.")
    elif health is None:
        st.error(f"Cannot reach the API server at {API_BASE}. Check that Railway is deployed and API_BASE_URL is set in Streamlit secrets.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            result = query_api(question.strip(), top_k)

        if result is None or "error" in result:
            detail = result.get("detail", "Unknown error") if result else "No response"
            st.error(f"Error: {detail}")

        else:
            # ── Strip LLM "Sources Used" footer from answer ─────────
            raw_answer = result.get("answer", "")
            clean_answer = re.split(
                r'\n+\*{0,2}(?:Sources? Used|References?|Sources?)\*{0,2}\s*[\:\-]?',
                raw_answer, flags=re.IGNORECASE
            )[0].strip()

            badge = confidence_badge(result.get("confidence", "low"))
            model = result.get("model", "")
            answerable = result.get("is_answerable", False)
            sources = result.get("sources", [])
            conf = result.get("confidence", "low")

            # ── Answer header row ────────────────────────────────────
            st.markdown("### Answer")
            meta_col1, meta_col2, meta_col3 = st.columns([3, 3, 2])
            with meta_col1:
                st.markdown(badge, unsafe_allow_html=True)
            with meta_col2:
                st.markdown(
                    f"<span style='font-size:12px;color:#888'>Model: {model}</span>",
                    unsafe_allow_html=True,
                )
            with meta_col3:
                icon = "✅" if answerable else "⚠️"
                label = "Answerable" if answerable else "Low confidence"
                st.markdown(
                    f"<span style='font-size:12px;color:#888'>{icon} {label}</span>",
                    unsafe_allow_html=True,
                )

            # ── Answer card (rendered markdown) ─────────────────────
            html_answer = md_lib.markdown(clean_answer, extensions=["extra"])
            st.markdown(
                f'<div class="answer-card">{html_answer}</div>',
                unsafe_allow_html=True,
            )

            # ── Quality indicator bar ────────────────────────────────
            verified_count = sum(1 for s in sources if s.get("verified"))
            conf_explain = {
                "high":   "strong semantic match (cosine similarity ≥ 0.65)",
                "medium": "moderate semantic match (cosine similarity 0.45–0.65)",
                "low":    "weak semantic match (cosine similarity < 0.45)",
            }.get(conf, "")
            halluc_val  = "✅ Passed" if answerable else "⚠️ Flagged"
            halluc_col  = "#2e7d32" if answerable else "#c62828"
            conf_col    = "#2e7d32" if conf == "high" else "#e65100" if conf == "medium" else "#c62828"

            st.markdown(f"""
<div style="display:flex;gap:12px;margin:12px 0 20px 0;">
  <div style="flex:1;background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
    <div style="font-size:11px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.5px">🔗 Citations Verified</div>
    <div style="font-size:26px;font-weight:700;color:#1a1a1a;margin-top:4px">{verified_count}/{len(sources)}</div>
    <div style="font-size:11px;color:#888;margin-top:2px">sources matched to chunks</div>
  </div>
  <div style="flex:1;background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
    <div style="font-size:11px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.5px">📊 Retrieval Confidence</div>
    <div style="font-size:26px;font-weight:700;color:{conf_col};margin-top:4px">{conf.upper()}</div>
    <div style="font-size:11px;color:#888;margin-top:2px">{conf_explain}</div>
  </div>
  <div style="flex:1;background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
    <div style="font-size:11px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.5px">🛡️ Hallucination Check</div>
    <div style="font-size:26px;font-weight:700;color:{halluc_col};margin-top:4px">{halluc_val}</div>
    <div style="font-size:11px;color:#888;margin-top:2px">two-layer resistance check</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Sources ──────────────────────────────────────────────
            if sources:
                st.markdown(
                    f"### Sources &nbsp;<span style='font-size:13px;color:#888;font-weight:400'>({len(sources)} retrieved)</span>",
                    unsafe_allow_html=True,
                )
                for i, src in enumerate(sources, 1):
                    pages = ", ".join(str(p) for p in src.get("page_numbers", []))
                    score = src.get("relevance_score", 0)
                    verified = src.get("verified", False)
                    score_color = "#2e7d32" if score >= 0.65 else "#e65100" if score >= 0.45 else "#c62828"
                    doc = src.get("document", "Unknown").replace("_", " ")
                    section = src.get("section", "")
                    verified_html = "<span style='color:#2e7d32;font-weight:600'>✅ Verified</span>" if verified else "<span style='color:#c62828;font-weight:600'>⚠️ Unverified</span>"
                    excerpt = src.get("text_excerpt", "").replace("<", "&lt;").replace(">", "&gt;")

                    st.markdown(f"""
<div style="background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:16px 20px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,0.06)">
  <div style="font-size:14px;font-weight:700;color:#d32f2f;margin-bottom:8px">{i}. {doc}</div>
  <div style="font-size:12px;color:#555;margin-bottom:10px">📌 {section}</div>
  <div style="display:flex;gap:24px;font-size:13px;color:#333;margin-bottom:12px">
    <span>📄 <b>Page(s):</b> {pages if pages else 'N/A'}</span>
    <span style="color:{score_color}"><b>▲ Relevance: {score:.2f}</b></span>
    <span>{verified_html}</span>
  </div>
  <div style="background:#f5f7ff;border-left:3px solid #c5cae9;border-radius:4px;padding:10px 14px;font-size:13px;color:#333;line-height:1.6">{excerpt}</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.info("No sources returned.")

            # ── Save to history ──────────────────────────────────────
            st.session_state.history.insert(0, {
                "question":  question.strip(),
                "answer":    clean_answer,
                "confidence": conf,
                "model":     model,
                "verified":  verified_count,
                "total":     len(sources),
            })


# ── Query History ───────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.markdown("### 🕓 Query History")
    for i, item in enumerate(st.session_state.history):
        conf_color = "#2e7d32" if item["confidence"] == "high" else "#e65100" if item["confidence"] == "medium" else "#c62828"
        with st.expander(f"Q: {item['question'][:80]}{'...' if len(item['question']) > 80 else ''}", expanded=False):
            st.markdown(
                f"<div style='font-size:11px;color:#888;margin-bottom:8px'>"
                f"<span style='color:{conf_color};font-weight:600'>{item['confidence'].upper()}</span> confidence &nbsp;·&nbsp; "
                f"{item['verified']}/{item['total']} verified &nbsp;·&nbsp; {item['model']}</div>",
                unsafe_allow_html=True,
            )
            html_hist = md_lib.markdown(item["answer"], extensions=["extra"])
            st.markdown(f'<div class="answer-card">{html_hist}</div>', unsafe_allow_html=True)
