# pdf_handler.py
import os
import json
import re
from dataclasses import dataclass
from typing import List, Sequence

import fitz  # PyMuPDF
import pandas as pd

from text_utils import normalize_text, clean_ocr_text
from ml_engine import train_term_filter, load_confirmed_terms, discover_terms_in_text

@dataclass
class TermHit:
    term: str
    page: int
    context: str
    tag: str
    score: float

def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    text = clean_ocr_text(text)
    blocks = [normalize_text(b) for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks:
        return blocks
    return [normalize_text(line) for line in text.split("\n") if line.strip()]

def find_context_for_term(paragraphs: Sequence[str], term: str) -> str:
    for p in paragraphs:
        if term in p:
            return p
    return ""

def save_terms_csv(term_hits: Sequence[TermHit], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = [
        {
            "term": h.term,
            "page": h.page,
            "context": h.context,
            "tag": h.tag,
            "score": round(h.score, 6),
        }
        for h in term_hits
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

def run_pdf_discovery_all_pages(
    csv_path: str,
    pdf_path: str,
    page_report_dir: str,
    merged_csv_path: str,
    confirmed_terms_path: str,
    column_name: str = "Japanese",
    limit: int = 99999,
    threshold: float = 0.72,
    use_ginza: bool = True,
):
    extra_terms = load_confirmed_terms(confirmed_terms_path)
    model, known_terms, metrics = train_term_filter(
        csv_path=csv_path,
        column_name=column_name,
        limit=limit,
        extra_positive_terms=extra_terms,
    )

    os.makedirs(page_report_dir, exist_ok=True)
    all_hits: List[TermHit] = []

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_text_raw = doc[page_idx].get_text("text")
        page_text = normalize_text(page_text_raw)
        paragraphs = split_paragraphs(page_text_raw)

        results = discover_terms_in_text(
            text=page_text,
            model=model,
            known_terms=known_terms,
            threshold=threshold,
            top_k=500,
            use_ginza=use_ginza,
        )

        page_payload = []
        for r in results:
            tag = "KNOWN" if r.is_known_dictionary_term else "NEW"
            context = find_context_for_term(paragraphs, r.candidate)
            hit = TermHit(
                term=r.candidate,
                page=page_num,
                context=context,
                tag=tag,
                score=r.score,
            )
            all_hits.append(hit)

            page_payload.append(
                {
                    "candidate": r.candidate,
                    "score": round(r.score, 6),
                    "tag": tag,
                    "context": context,
                }
            )

        out_json = os.path.join(page_report_dir, f"page_{page_num:03d}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "page": page_num,
                    "hits": page_payload,
                    "count": len(page_payload),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    doc.close()
    save_terms_csv(all_hits, merged_csv_path)
    return metrics, all_hits, total_pages