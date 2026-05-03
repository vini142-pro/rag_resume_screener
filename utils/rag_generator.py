# utils/rag_generator.py
# ─────────────────────────────────────────────────────────────────────
#  RAG Generator — the "Generation" part of RAG
#
#  Takes retrieved resume chunks + job description as context,
#  then uses a HuggingFace text-generation model to produce
#  a natural-language recruiter summary report.
#
#  Model: google/flan-t5-base  (free, local, ~250MB, instruction-tuned)
# ─────────────────────────────────────────────────────────────────────

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

GENERATOR_MODEL = "google/flan-t5-base"


class RAGGenerator:
    """
    Generation module: given retrieved context (resumes) + query (JD),
    generates a structured recruiter summary using an LLM.
    """

    def __init__(self, model_name: str = GENERATOR_MODEL):
        print(f"  🤖  Loading generator model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe      = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            do_sample=False,         # deterministic output
        )
        print(f"  ✅  Generator model loaded.")

    def _build_prompt(self, job_role: str, job_description: str,
                      retrieved_resumes: list[dict]) -> str:
        """
        Construct a RAG prompt:
          [Context] = top retrieved resume snippets
          [Query]   = job description
          [Task]    = generate shortlist recommendation
        """
        context_parts = []
        for i, r in enumerate(retrieved_resumes, 1):
            snippet = r["text"][:400].strip().replace("\n", " ")
            score   = r.get("similarity_score", 0) * 100
            context_parts.append(
                f"Candidate {i}: {r['candidate']} (Similarity: {score:.1f}%)\n{snippet}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""You are an AI recruiter assistant. Based on the retrieved candidate profiles below, 
write a professional shortlist recommendation for the role of {job_role}.

Job Requirements Summary:
{job_description[:300].strip()}

Retrieved Candidate Profiles:
{context}

Task: Summarize which candidates are most suitable, mention their key strengths, and recommend the top 3.
Recommendation:"""

        return prompt

    def generate(self, job_role: str, job_description: str,
                 retrieved_resumes: list[dict]) -> str:
        """
        Full RAG generation pipeline:
          1. Build prompt from context + query
          2. Generate response via LLM
          3. Return clean recruiter summary
        """
        prompt = self._build_prompt(job_role, job_description, retrieved_resumes)

        # Truncate prompt to model max input (512 tokens for flan-t5-base)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        truncated_prompt = self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True
        )

        output = self.pipe(truncated_prompt)[0]["generated_text"]
        return output.strip()


def generate_rule_based_summary(job_role: str, retrieved_resumes: list[dict]) -> str:
    """
    Fallback: Rule-based summary generator (no model needed).
    Used when LLM generation is skipped with --no-llm flag.
    """
    lines = [f"\n  📋  AI Recruiter Summary — {job_role}", "  " + "─" * 55]

    for i, r in enumerate(retrieved_resumes, 1):
        score = r.get("similarity_score", 0) * 100
        medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"#{i}"
        verdict = "Strong fit" if score >= 70 else "Moderate fit" if score >= 50 else "Weak fit"

        # Extract first non-empty line as headline
        headline = next(
            (ln.strip() for ln in r["text"].split("\n") if ln.strip()), r["candidate"]
        )
        lines.append(f"\n  {medal}  {r['candidate']}  |  Score: {score:.1f}%  |  {verdict}")
        lines.append(f"       {textwrap.shorten(headline, width=70)}")

    lines.append(f"\n  💡  Recommendation: Shortlist top 3 candidates for interview.")
    return "\n".join(lines)
