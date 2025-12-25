from typing import List, Tuple



def word_overlap_chunks(text: str, target_words=200, overlap_words=50) -> List[Tuple[int, int, str]]:
    words = text.split()
    if target_words <= 0:
        raise ValueError("target_words must be > 0")
    step = target_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be < target_words")
    chunks = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        chunks.append((start, min(end, len(words)), chunk_text))
        start += step
        if start >= len(words):
            break
    return chunks