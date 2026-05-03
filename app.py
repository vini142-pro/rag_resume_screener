# app.py
# ═══════════════════════════════════════════════════════════════════════
#  RAG Resume Screener — Streamlit Web App

#  Deploy: streamlit run app.py
# ═══════════════════════════════════════════════════════════════════════

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .hero-sub { text-align:center; color:#8b8fa8; font-size:1rem; margin-bottom:1.5rem; }
    .pipeline-badge {
        display:inline-block; background:linear-gradient(135deg,#667eea22,#764ba222);
        border:1px solid #667eea55; border-radius:20px;
        padding:4px 14px; font-size:0.82rem; color:#a78bfa; margin:2px;
    }
    .metric-card {
        background:linear-gradient(135deg,#1e1e2e,#2a2a3e);
        border:1px solid #333355; border-radius:12px;
        padding:1.2rem; text-align:center;
    }
    .rank-card {
        background:linear-gradient(135deg,#1a1a2e,#16213e);
        border-left:4px solid #667eea; border-radius:8px;
        padding:1rem 1.2rem; margin-bottom:0.8rem;
    }
    .rank-card-gold   { border-left-color:#FFD700; }
    .rank-card-silver { border-left-color:#C0C0C0; }
    .rank-card-bronze { border-left-color:#CD7F32; }
    .score-pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:bold; }
    .score-high   { background:#1a472a; color:#4ade80; }
    .score-medium { background:#3d2b00; color:#fbbf24; }
    .score-low    { background:#3b1515; color:#f87171; }
    .skill-tag {
        display:inline-block; background:#1e3a5f; color:#60a5fa;
        border-radius:4px; padding:2px 8px; font-size:0.75rem; margin:2px;
    }
    .missing-tag {
        display:inline-block; background:#3b1515; color:#f87171;
        border-radius:4px; padding:2px 8px; font-size:0.75rem; margin:2px;
    }
    .section-header {
        font-size:1.3rem; font-weight:700; color:#e2e8f0;
        border-bottom:2px solid #667eea44; padding-bottom:0.4rem; margin:1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Cached model loaders
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedder():
    from embeddings.embedder import Embedder
    return Embedder()

@st.cache_resource(show_spinner=False)
def load_store():
    from embeddings.vector_store import FAISSVectorStore
    from data.sample_data import RESUMES
    from utils.preprocessor import extract_skills
    embedder = load_embedder()
    store = FAISSVectorStore(dim=embedder.model.get_sentence_embedding_dimension())
    if store.is_saved():
        store.load()
    else:
        texts = [r["text"] for r in RESUMES]
        embs  = embedder.embed(texts)
        meta  = [{**r, "skills": extract_skills(r["text"])} for r in RESUMES]
        store.build_index(embs, meta)
        store.save()
    return store


def retrieve(jd_text, top_k=5):
    emb = load_embedder().embed_single(jd_text)
    return load_store().search(emb, top_k=top_k)

def score_class(s):
    return "score-high" if s >= .70 else "score-medium" if s >= .50 else "score-low"

def score_label(s):
    return "Strong Fit ✅" if s >= .70 else "Moderate Fit 🔶" if s >= .50 else "Weak Fit ❌"

MEDALS = ["🥇","🥈","🥉"]
RANK_CLS = ["rank-card-gold","rank-card-silver","rank-card-bronze"]


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Top Candidates to Retrieve", 3, 10, 5)
    st.markdown("---")
    st.markdown("## 🧠 RAG Pipeline")
    for icon, step, tool in [
        ("1️⃣","Embed",    "HuggingFace MiniLM"),
        ("2️⃣","Store",    "FAISS Vector Index"),
        ("3️⃣","Retrieve", "Cosine Similarity"),
        ("4️⃣","Generate", "Rule-based LLM"),
    ]:
        st.markdown(f"**{icon} {step}** — `{tool}`")
    st.markdown("---")
    st.markdown("## 📦 Tech Stack")
    for t in ["sentence-transformers","faiss-cpu","HuggingFace","Streamlit"]:
        st.markdown(f'<span class="pipeline-badge">{t}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Future Interns — ML Internship 2026")


# ══════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">🔍 RAG Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Retrieval-Augmented Generation · HuggingFace Embeddings · FAISS Vector Search</div>', unsafe_allow_html=True)

_, mid, _ = st.columns([1,6,1])
with mid:
    steps = ["📄 Resume Input","→","🧠 Embed","→","📦 FAISS","→","🔍 Retrieve","→","📋 AI Report"]
    st.markdown(" ".join(f'<span class="pipeline-badge">{b}</span>' for b in steps), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("🤖 Loading HuggingFace model & building FAISS index..."):
    try:
        load_embedder(); load_store()
        st.success("✅ Models loaded & FAISS index ready!", icon="🚀")
    except Exception as e:
        st.error(f"❌ Model load failed: {e}"); st.stop()


# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Screen Candidates",
    "📊 Analytics Dashboard",
    "📝 Screen Custom Resume",
    "🏗️ About RAG",
])


# ─── TAB 1: Screen Candidates ─────────────────────────────────────────
with tab1:
    from data.sample_data import JOB_DESCRIPTIONS
    from utils.preprocessor import extract_skills
    from utils.rag_generator import generate_rule_based_summary

    st.markdown('<div class="section-header">📋 Job Description Input</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1,2])
    with col_l:
        preset = st.selectbox("Preset Role", ["— Custom JD —"] + list(JOB_DESCRIPTIONS.keys()))
    jd_val   = JOB_DESCRIPTIONS.get(preset, "")
    role_val = preset if preset != "— Custom JD —" else "Custom Role"
    jd_input = st.text_area("Job Description", value=jd_val, height=180,
                             placeholder="Paste job description here...", label_visibility="collapsed")
    role_name = st.text_input("Role Name", value=role_val)

    if st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True):
        if not jd_input.strip():
            st.warning("Please enter a job description.")
        else:
            with st.spinner("🔍 Retrieving top candidates via FAISS..."):
                results   = retrieve(jd_input, top_k=top_k)
                jd_skills = extract_skills(jd_input)

            st.markdown(f'<div class="section-header">🏆 Top {top_k} Candidates — {role_name}</div>', unsafe_allow_html=True)

            for i, r in enumerate(results):
                s       = r["similarity_score"]
                pct     = s * 100
                medal   = MEDALS[i] if i < 3 else f"#{i+1}"
                rcls    = RANK_CLS[i] if i < 3 else "rank-card"
                matched = set(r.get("skills",[])) & set(jd_skills)
                missing = set(jd_skills) - set(r.get("skills",[]))
                mh = " ".join(f'<span class="skill-tag">{x}</span>' for x in sorted(matched))
                xh = " ".join(f'<span class="missing-tag">{x}</span>' for x in sorted(missing))

                st.markdown(f"""
                <div class="rank-card {rcls}">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{medal} &nbsp; {r['candidate']}</span>
                    <span>
                      <span class="score-pill {score_class(s)}">{score_label(s)}</span>
                      &nbsp;<span style="color:#94a3b8;font-size:0.9rem;">{pct:.1f}% match</span>
                    </span>
                  </div>
                  <div style="margin-top:0.5rem;">
                    <span style="color:#64748b;font-size:0.8rem;">✅ Matched: </span>
                    {mh if mh else '<span style="color:#4ade80;font-size:0.8rem;">All matched!</span>'}
                  </div>
                  {'<div style="margin-top:0.3rem;"><span style="color:#64748b;font-size:0.8rem;">❌ Missing: </span>' + xh + '</div>' if missing else ''}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">📊 Match Scores</div>', unsafe_allow_html=True)
            for r in results:
                c1, c2 = st.columns([2,5])
                with c1: st.markdown(f"**{r['candidate']}**")
                with c2: st.progress(r["similarity_score"], text=f"{r['similarity_score']*100:.1f}%")

            st.markdown('<div class="section-header">✍️ AI Recruiter Summary</div>', unsafe_allow_html=True)
            st.code(generate_rule_based_summary(role_name, results), language=None)

            st.markdown('<div class="section-header">📈 Similarity Chart</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,4))
            fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1a2e")
            names  = [r["candidate"] for r in results]
            scores = [r["similarity_score"]*100 for r in results]
            colors = ["#4ade80" if s>=70 else "#fbbf24" if s>=50 else "#f87171" for s in scores]
            bars   = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.55)
            for bar, sc in zip(bars, scores[::-1]):
                ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                        f"{sc:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")
            ax.axvline(60, color="#f87171", linestyle="--", lw=1, alpha=0.6, label="60% threshold")
            ax.set_xlim(0,108)
            ax.tick_params(colors="white"); ax.spines[["top","right","bottom"]].set_visible(False)
            ax.spines["left"].set_color("#333355"); ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ─── TAB 2: Analytics Dashboard ───────────────────────────────────────
with tab2:
    from data.sample_data import JOB_DESCRIPTIONS, RESUMES
    from utils.preprocessor import extract_skills

    st.markdown('<div class="section-header">📊 Full Analytics — All Roles × All Candidates</div>', unsafe_allow_html=True)

    if st.button("🔄 Generate Full Dashboard", type="primary"):
        with st.spinner("Running retrieval for all roles..."):
            all_res = {role: retrieve(jd, top_k=len(RESUMES)) for role, jd in JOB_DESCRIPTIONS.items()}

        all_cands  = [r["candidate"] for r in RESUMES]
        roles      = list(all_res.keys())
        cand_idx   = {c:i for i,c in enumerate(all_cands)}
        matrix     = np.zeros((len(roles), len(all_cands)))
        for ri,(role,results) in enumerate(all_res.items()):
            for r in results:
                ci = cand_idx.get(r["candidate"])
                if ci is not None: matrix[ri][ci] = r["similarity_score"]*100

        st.markdown("#### 🌡️ Semantic Similarity Heatmap")
        fig, ax = plt.subplots(figsize=(12,3.5))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1a2e")
        sns.heatmap(matrix, ax=ax, xticklabels=all_cands, yticklabels=roles,
                    annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
                    linewidths=0.5, linecolor="#1a1a2e")
        ax.tick_params(colors="white", labelsize=8)
        plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("#### 📋 Best Candidate per Role")
        rows = [{"Role": role, "Best Candidate": r[0]["candidate"],
                 "Score": f"{r[0]['similarity_score']*100:.1f}%",
                 "Verdict": score_label(r[0]["similarity_score"])}
                for role, r in all_res.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🛠️ Skills per Candidate")
        store = load_store()
        skill_rows = [{"Candidate": m["candidate"],
                       "Total Skills": len(m.get("skills",[])),
                       "Skills": ", ".join(sorted(m.get("skills",[])))
                      } for m in store.metadata]
        st.dataframe(pd.DataFrame(skill_rows), use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click the button to generate the full multi-role analytics dashboard.")


# ─── TAB 3: Screen Custom Resume ──────────────────────────────────────
with tab3:
    from data.sample_data import JOB_DESCRIPTIONS
    from utils.preprocessor import extract_skills

    st.markdown('<div class="section-header">📝 Screen Your Own Resume</div>', unsafe_allow_html=True)
    st.markdown("Paste your resume and pick a target role to get an instant match score.")

    col_a, col_b = st.columns(2)
    with col_a:
        resume_txt = st.text_area("Your Resume Text", height=300,
                                   placeholder="Paste your full resume here...")
    with col_b:
        t_role = st.selectbox("Target Role", list(JOB_DESCRIPTIONS.keys()))
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Screen My Resume", type="primary", use_container_width=True):
            if not resume_txt.strip():
                st.warning("Please paste your resume text.")
            else:
                with st.spinner("Analyzing..."):
                    emb   = load_embedder()
                    r_emb = emb.embed_single(resume_txt)
                    j_emb = emb.embed_single(JOB_DESCRIPTIONS[t_role])
                    sim   = float(np.dot(r_emb, j_emb))
                    jd_sk = set(extract_skills(JOB_DESCRIPTIONS[t_role]))
                    rs_sk = set(extract_skills(resume_txt))
                    match = rs_sk & jd_sk
                    miss  = jd_sk - rs_sk

                st.metric("Semantic Match Score", f"{sim*100:.1f}%")
                st.progress(sim)
                color = "#4ade80" if sim >= .70 else "#fbbf24" if sim >= .50 else "#f87171"
                verdict = "✅ SHORTLISTED" if sim >= .60 else "❌ NOT SHORTLISTED"
                st.markdown(f'<h3 style="color:{color};text-align:center;">{verdict}</h3>', unsafe_allow_html=True)

                st.markdown("**✅ Skills you have:**")
                if match:
                    st.markdown(" ".join(f'<span class="skill-tag">{s}</span>' for s in sorted(match)), unsafe_allow_html=True)
                else:
                    st.warning("No matching skills detected.")

                st.markdown("**❌ Skills to add:**")
                if miss:
                    st.markdown(" ".join(f'<span class="missing-tag">{s}</span>' for s in sorted(miss)), unsafe_allow_html=True)
                else:
                    st.success("You have all the required skills! 🌟")


# ─── TAB 4: About RAG ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🏗️ How the RAG Pipeline Works</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, icon, title, desc, tool in zip(
        [c1,c2,c3,c4],
        ["🧠","📦","🔍","✍️"],
        ["Embed","Store","Retrieve","Generate"],
        [
            "Resumes + JDs converted to 384-dim semantic vectors",
            "Vectors stored in FAISS for millisecond-speed search",
            "JD vector compared to all resumes via cosine similarity",
            "Top-K resumes used as context to generate recruiter report",
        ],
        ["all-MiniLM-L6-v2","faiss-cpu","Cosine Similarity","Rule-based LLM"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-weight:700;color:#e2e8f0;margin:0.4rem 0;">{title}</div>
              <div style="color:#8b8fa8;font-size:0.8rem;">{desc}</div>
              <div style="margin-top:0.6rem;"><span class="pipeline-badge">{tool}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚡ RAG vs Traditional Matching</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Feature":              ["Matching Type","Understands Synonyms","Context Aware","Generates Summary","Retraining Needed"],
        "Traditional (TF-IDF)": ["Keyword","❌","❌","❌","✅ Yes"],
        "This RAG System":      ["Semantic","✅","✅","✅","❌ No"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">📁 Project Structure</div>', unsafe_allow_html=True)
    st.code("""
rag_resume_screener/
├── app.py                 ← Streamlit web app (this file)
├── main.py                ← CLI entry point
├── requirements.txt
├── data/
│   ├── sample_data.py     ← 10 resumes + 3 job descriptions
│   └── sample_resume.txt
├── embeddings/
│   ├── embedder.py        ← HuggingFace sentence transformer
│   └── vector_store.py    ← FAISS index
└── utils/
    ├── preprocessor.py    ← Text cleaning + skill extraction
    ├── rag_generator.py   ← LLM generation module
    └── visualizer.py      ← Chart generation
    """, language="text")