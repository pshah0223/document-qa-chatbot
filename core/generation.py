import re
from typing import List, Tuple, Dict
from transformers import pipeline



_flan_pipe = None
def get_flan_pipe():
    global _flan_pipe
    if _flan_pipe is None:
        _flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=120, do_sample=False)
    return _flan_pipe


def build_prompt(query: str, contexts: List[str]) -> str:
    ctx = "\n\n---\n\n".join([f"[Source {i+1}]\n{c}" for i,c in enumerate(contexts)])
    return f"""Answer the question ONLY from the context below.\nIf the answer is not present, say \"I don't know\".\nBe concise (1-2 sentences).\n\nContext:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"""


def generate_answer(query: str, hits: List[Tuple[float, Dict]], min_score=0.0, max_contexts=6) -> str:
    filtered = [h for h in hits if h[0] >= min_score]
    if not filtered:
        return "I don't know."
    contexts = [h[1]["preview"] if len(h[1]["preview"])<1000 else h[1]["preview"][:1000] for h in filtered[:max_contexts]]
    prompt = build_prompt(query, contexts)
    pipe = get_flan_pipe()
    out = pipe(prompt)[0]["generated_text"].strip()
    return re.sub(r"\s+", " ", out)