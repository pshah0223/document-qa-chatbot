import re
import io
import os
import pdfplumber
import docx
import pandas as pd
from typing import List, Dict, Tuple





def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"-\n(?=[a-z])", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s



def _table_to_sentences(table: List[List[str]]) -> List[str]:
    if not table or len(table) < 2:
        return []
    header = [(c.strip() if c else f"col{i}") for i, c in enumerate(table[0])]
    rows = []
    for r in table[1:]:
        r = [(c.strip() if c else "") for c in r]
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
        rows.append(r)
    try:
        df = pd.DataFrame(rows, columns=header)
        sentences = []
        for _, row in df.iterrows():
            if len(df.columns) >= 2 and str(row.iloc[0]).isdigit() and str(row.iloc[1]).isdigit():
                sentences.append(f"In a {int(row.iloc[0])}-credit course, students are allowed {int(row.iloc[1])} absences.")
                continue
            parts = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val and val.lower() not in ("nan", "none"):
                    parts.append(f"{col}: {val}")
            if parts:
                sentences.append("; ".join(parts) + ".")
        return sentences
    except Exception:
        out = []
        for r in rows:
            parts = [f"{header[i]}: {r[i]}" for i in range(len(header)) if r[i]]
            if parts:
                out.append("; ".join(parts) + ".")
        return out
    




def extract_pdf_bytes(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = _clean_text(page.extract_text() or "")
            row_sents = []
            try:
                tables = page.extract_tables()
                for t in tables or []:
                    row_sents.extend(_table_to_sentences(t))
            except Exception:
                pass
            if row_sents:
                txt = (txt + "\n\n" + "\n".join(row_sents)).strip()
            pages.append({"page": i, "text": txt})
    all_text = "\n\n".join([_clean_text(p["text"]) for p in pages if p["text"].strip()])
    return all_text, pages



def extract_docx_bytes(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs]
    paras = [_clean_text(p) for p in paras if _clean_text(p)]
    text = "\n\n".join(paras)
    return text, [{"page": 1, "text": text}]



def extract_any_bytes(filename: str, file_bytes: bytes) -> Tuple[str, List[Dict]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pdf_bytes(file_bytes)
    elif ext == ".docx":
        return extract_docx_bytes(file_bytes)
    else:
        raise ValueError("Unsupported file type: " + ext)