from collections import Counter
from typing import Callable, Optional
import re
import json
import base64

def is_word_order_shuffled(student_tokens: list[str], reference_tokens: list[str]) -> bool:
    return Counter(student_tokens) == Counter(reference_tokens) and student_tokens != reference_tokens

EXACT_MATCH_SIM_THRESHOLD = 0.995

def check_exact_match_override(
    student_clean_text: str,
    references_clean: list[str],
    cosine_sim_fn: Callable[[str, str], float],
) -> Optional[dict]:
    for ref in references_clean:
        if student_clean_text == ref or cosine_sim_fn(student_clean_text, ref) >= EXACT_MATCH_SIM_THRESHOLD:
            return {
                "label": "correct",
                "confidence": 0.99,
                "evidence": "Matches a reference answer almost word-for-word.",
                "override": "exact_match_shortcircuit",
            }
    return None

def extract_qa_with_tesseract(image_bytes: bytes) -> list[dict]:
    import io
    import pytesseract
    from PIL import Image
    raw_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
    return parse_numbered_text(raw_text)

def parse_numbered_text(raw_text: str) -> list[dict]:
    QUESTION_NUM_PATTERN = re.compile(r"^\s*(\d{1,2})[\.\)]\s*", re.MULTILINE)
    matches = list(QUESTION_NUM_PATTERN.finditer(raw_text))
    results = []
    for i, m in enumerate(matches):
        start, number = m.end(), int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        results.append({"number": number, "question_text": None, "answer_text": raw_text[start:end].strip()})
    return results
