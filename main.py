# main.py
# ═══════════════════════════════════════════════════════════════════════
#  RAG Resume Screener
#
#  PIPELINE:
#    1. Embed resumes → HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
#    2. Store vectors → FAISS (local, no server needed)
#    3. Retrieve top-K → Semantic similarity search
#    4. Generate report → Flan-T5 LLM (free, local)
#
#  HOW TO RUN:
#    python main.py                          → full RAG pipeline (all roles)
#    python main.py --role "NLP Engineer"   → single role
#    python main.py --no-llm                → skip LLM, use rule-based summary
#    python main.py --top-k 3               → retrieve top 3 resumes per role
#    python main.py --rebuild               → force re-embed (ignore saved index)
#    python main.py --resume "path/to/resume.txt" --role "NLP Engineer"
#
#  OUTPUT:
#    Terminal : ranked tables + AI-generated recruiter summaries
#    outputs/ : 4 PNG charts saved automatically
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tabulate import tabulate

from data.sample_data       import RESUMES, JOB_DESCRIPTIONS
from embeddings.embedder    import Embedder
from embeddings.vector_store import FAISSVectorStore
from utils.preprocessor     import extract_skills
from utils.visualizer       import (
    plot_similarity_rankings,
    plot_similarity_heatmap,
    plot_skill_breakdown,
    plot_rag_pipeline_diagram,
)
from utils.rag_generator    import RAGGenerator, generate_rule_based_summary

os.makedirs("outputs", exist_ok=True)


# ════════════════════════════════════════════════════════════
#  STEP 1 — Indexing (Embed + Store in FAISS)
# ════════════════════════════════════════════════════════════

def build_index(embedder: Embedder, rebuild: bool = False) -> FAISSVectorStore:
    store = FAISSVectorStore(dim=embedder.model.get_sentence_embedding_dimension())

    if store.is_saved() and not rebuild:
        print("\n📂  Loading saved FAISS index (use --rebuild to re-embed)...")
        store.load()
        return store

    print("\n🔨  Building FAISS index from scratch...")
    texts = [r["text"] for r in RESUMES]
    print(f"  📝  Embedding {len(texts)} resumes via HuggingFace...")
    embeddings = embedder.embed(texts)

    # Attach skills to metadata for later analysis
    metadata = []
    for r in RESUMES:
        meta = dict(r)
        meta["skills"] = extract_skills(r["text"])
        metadata.append(meta)

    store.build_index(embeddings, metadata)
    store.save()
    return store


# ════════════════════════════════════════════════════════════
#  STEP 2 — Retrieval (Semantic Search per Job Role)
# ════════════════════════════════════════════════════════════

def retrieve_top_k(embedder: Embedder, store: FAISSVectorStore,
                   job_descriptions: dict, top_k: int = 5) -> dict:
    print(f"\n🔍  Retrieving top-{top_k} candidates per role via semantic search...")
    all_results = {}
    for role, jd in job_descriptions.items():
        query_emb = embedder.embed_single(jd)
        results   = store.search(query_emb, top_k=top_k)
        all_results[role] = results
        print(f"  ✅  Retrieved for: {role}")
    return all_results


# ════════════════════════════════════════════════════════════
#  STEP 3 — Display Rankings Table
# ════════════════════════════════════════════════════════════

def display_rankings(all_results: dict):
    print("\n\n" + "═" * 65)
    print("   📊  SEMANTIC SIMILARITY RANKINGS (RAG Retrieval)")
    print("═" * 65)

    for role, results in all_results.items():
        print(f"\n{'─'*65}")
        print(f"  🎯  {role}")
        print(f"{'─'*65}")

        table = []
        for i, r in enumerate(results, 1):
            score    = r["similarity_score"] * 100
            medal    = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"#{i}"
            n_skills = len(r.get("skills", []))
            verdict  = "✅ Strong" if score >= 70 else "🔶 Moderate" if score >= 50 else "❌ Weak"
            table.append([medal, r["candidate"], f"{score:.1f}%", n_skills, verdict])

        print(tabulate(
            table,
            headers=["Rank", "Candidate", "Similarity", "Skills Found", "Fit"],
            tablefmt="fancy_grid"
        ))


# ════════════════════════════════════════════════════════════
#  STEP 4 — Skill Gap Analysis
# ════════════════════════════════════════════════════════════

def display_skill_gaps(all_results: dict, jd_skills: dict):
    print("\n\n" + "─" * 65)
    print("  🔍  SKILL GAP ANALYSIS")
    print("─" * 65)

    for role, results in all_results.items():
        required = set(jd_skills[role])
        print(f"\n  📋  {role}")
        print(f"  Required Skills ({len(required)}): {sorted(required)}")

        for i, r in enumerate(results[:5], 1):
            matched = set(r.get("skills", [])) & required
            missing = required - set(r.get("skills", []))
            score   = r["similarity_score"] * 100
            print(f"\n  #{i}  {r['candidate']}  (Semantic Score: {score:.1f}%)")
            print(f"       ✅ Has     ({len(matched)}): {sorted(matched)}")
            if missing:
                print(f"       ❌ Missing ({len(missing)}): {sorted(missing)}")
            else:
                print(f"       🌟 Full skill match!")


# ════════════════════════════════════════════════════════════
#  STEP 5 — RAG Generation (LLM Recruiter Summary)
# ════════════════════════════════════════════════════════════

def generate_reports(all_results: dict, job_descriptions: dict,
                     generator=None, use_llm: bool = True):
    print("\n\n" + "═" * 65)
    print("   ✍️   RAG GENERATION — AI Recruiter Summaries")
    print("═" * 65)

    for role, results in all_results.items():
        print(f"\n{'─'*65}")
        print(f"  🎯  {role}")
        print(f"{'─'*65}")

        if use_llm and generator:
            print("  ⏳  Generating with LLM...")
            summary = generator.generate(role, job_descriptions[role], results)
            print(f"\n  📋  LLM Summary:\n")
            # Wrap long output
            import textwrap
            for line in summary.split("\n"):
                print("  " + textwrap.fill(line, width=65))
        else:
            # Rule-based fallback
            print(generate_rule_based_summary(role, results))


# ════════════════════════════════════════════════════════════
#  STEP 6 — Screen a Single Custom Resume
# ════════════════════════════════════════════════════════════

def screen_custom_resume(resume_path: str, target_role: str,
                         embedder: Embedder, store: FAISSVectorStore,
                         job_descriptions: dict):
    print(f"\n{'═'*60}")
    print(f"  🔎  Screening Custom Resume → {target_role}")
    print(f"{'═'*60}")

    if not os.path.exists(resume_path):
        print(f"❌  File not found: {resume_path}")
        sys.exit(1)

    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    # Embed the custom resume
    resume_emb = embedder.embed_single(resume_text)

    # Embed the job description and compute cosine similarity
    jd_text = job_descriptions.get(target_role, "")
    if not jd_text:
        print(f"❌  Role '{target_role}' not found. Available: {list(job_descriptions.keys())}")
        sys.exit(1)

    jd_emb = embedder.embed_single(jd_text)
    similarity = float(np.dot(resume_emb, jd_emb))  # L2-normalized → cosine

    skills_found = extract_skills(resume_text)
    jd_skills    = extract_skills(jd_text)
    matched      = set(skills_found) & set(jd_skills)
    missing      = set(jd_skills) - set(skills_found)

    print(f"  Semantic Similarity : {similarity * 100:.1f}%")
    print(f"  Skills Found        : {sorted(skills_found)}")
    print(f"  Skills Matched      : {sorted(matched)}")
    print(f"  Skills Missing      : {sorted(missing) if missing else 'None!'}")
    verdict = "✅  SHORTLISTED" if similarity >= 0.60 else "❌  NOT SHORTLISTED"
    print(f"\n  Verdict → {verdict}")


# ════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RAG Resume Screener — HuggingFace + FAISS + LLM"
    )
    p.add_argument("--role",    type=str,   default=None,
                   help="Filter to a single job role")
    p.add_argument("--resume",  type=str,   default=None,
                   help="Path to a .txt resume to screen")
    p.add_argument("--top-k",   type=int,   default=5,
                   help="Number of top candidates to retrieve (default: 5)")
    p.add_argument("--no-llm",  action="store_true",
                   help="Skip LLM generation, use rule-based summary instead")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip chart generation")
    p.add_argument("--rebuild", action="store_true",
                   help="Force re-embed resumes (ignore saved FAISS index)")
    return p.parse_args()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("\n" + "═" * 65)
    print("   🔍  RAG Resume Screener")
    print("   HuggingFace Embeddings + FAISS + LLM Generation")
    print("═" * 65)

    # ── Load models ──────────────────────────────────────────
    print("\n⚙️   Loading models...")
    embedder  = Embedder()
    generator = None
    if not args.no_llm:
        try:
            generator = RAGGenerator()
        except Exception as e:
            print(f"  ⚠️  LLM loading failed ({e}). Falling back to rule-based summaries.")
            args.no_llm = True

    # ── Step 1: Index ─────────────────────────────────────────
    store = build_index(embedder, rebuild=args.rebuild)

    # ── Single resume mode ────────────────────────────────────
    if args.resume:
        if not args.role:
            print("❌  Please specify --role with --resume.")
            sys.exit(1)
        screen_custom_resume(args.resume, args.role, embedder, store, JOB_DESCRIPTIONS)
        return

    # ── Select roles ──────────────────────────────────────────
    job_descriptions = JOB_DESCRIPTIONS
    if args.role:
        if args.role not in JOB_DESCRIPTIONS:
            print(f"❌  Role '{args.role}' not found. Available: {list(JOB_DESCRIPTIONS.keys())}")
            sys.exit(1)
        job_descriptions = {args.role: JOB_DESCRIPTIONS[args.role]}

    # ── Step 2: Retrieve ──────────────────────────────────────
    all_results = retrieve_top_k(embedder, store, job_descriptions, top_k=args.top_k)

    # ── Step 3: Display rankings ──────────────────────────────
    display_rankings(all_results)

    # ── Step 4: Skill gap analysis ────────────────────────────
    jd_skills = {role: extract_skills(jd) for role, jd in job_descriptions.items()}
    display_skill_gaps(all_results, jd_skills)

    # ── Step 5: RAG generation ────────────────────────────────
    generate_reports(all_results, job_descriptions, generator, use_llm=not args.no_llm)

    # ── Step 6: Visualizations ────────────────────────────────
    if not args.no_plots:
        print("\n\n📈  Generating visualizations → outputs/")
        all_candidates = [r["candidate"] for r in RESUMES]
        plot_similarity_rankings(all_results)
        plot_similarity_heatmap(all_results, all_candidates)
        plot_skill_breakdown(all_results, jd_skills)
        plot_rag_pipeline_diagram()
        print("  ✅  All charts saved to outputs/\n")

    print("\n✅  RAG Pipeline Complete!\n")


if __name__ == "__main__":
    main()
