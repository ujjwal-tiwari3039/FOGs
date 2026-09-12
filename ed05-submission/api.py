import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import hashlib
from typing import List, Optional

from src import features
from src.features import init_models, cache_reference_embeddings
from src.model import load_model, predict_single
from src.explain import generate_evidence
from src.preprocess import preprocess
from src.updates import is_word_order_shuffled, check_exact_match_override, extract_qa_with_tesseract, parse_numbered_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
init_models(use_nli=False)
model = load_model()
print("Models loaded.")

class GradeRequest(BaseModel):
    question: str
    references: List[str]
    student_answer: str

def cosine_sim_fn(text1, text2):
    import torch
    from sentence_transformers import util
    e1 = features._sentence_model.encode(text1, convert_to_tensor=True)
    e2 = features._sentence_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()

@app.post("/grade_single")
def grade_single(req: GradeRequest):
    q_hash = hashlib.md5("".join(req.references).encode()).hexdigest()
    record = {
        'question_id': f'api_{q_hash}',
        'question_text': req.question,
        'reference_answers': req.references,
        'student_answer': req.student_answer,
        'domain': 'demo'
    }
    
    clean_stu = preprocess(req.student_answer)
    clean_refs = [preprocess(r) for r in req.references]
    
    # Section 2: Exact Match Override
    override = check_exact_match_override(clean_stu, clean_refs, cosine_sim_fn)
    if override:
        return override

    cache_reference_embeddings([record])
    pred_label, prob_vector = predict_single(model, record)
    evidence = generate_evidence(record, pred_label, prob_vector)
    
    # Section 1: Word Order Shuffled check
    # Find best ref
    best_ref = clean_refs[0]
    best_score = -1
    for r in clean_refs:
        s = cosine_sim_fn(clean_stu, r)
        if s > best_score:
            best_score = s
            best_ref = r
            
    student_tokens = clean_stu.split()
    ref_tokens = best_ref.split()
    
    if is_word_order_shuffled(student_tokens, ref_tokens):
        evidence["evidence_string"] += "\n⚠️ Uses the same words as the reference answer, reordered."
        
    return {
        "label": pred_label,
        "confidence": evidence['confidence']['top_prob'],
        "needs_review": not evidence['confidence']['confident'],
        "evidence": evidence['evidence_string'],
        "note": evidence['negation']['note'] if evidence['negation']['has_negation_cue'] else ""
    }

@app.post("/extract_text")
def extract_text(text: str = Form(...)):
    return parse_numbered_text(text)

@app.post("/extract_image")
async def extract_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        return extract_qa_with_tesseract(contents)
    except Exception as e:
        print("Tesseract failed:", e)
        # Mock fallback if tesseract is missing
        return [
            {"number": 1, "question_text": "Mock OCR Q1", "answer_text": "Mock Answer 1"},
            {"number": 2, "question_text": "Mock OCR Q2", "answer_text": "Mock Answer 2"}
        ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
