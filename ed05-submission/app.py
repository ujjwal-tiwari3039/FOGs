import streamlit as st
import pandas as pd
import hashlib
import re
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features import init_models, cache_reference_embeddings
from src.model import load_model, predict_single
from src.explain import generate_evidence

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ED-05 Grader", page_icon="🎓", layout="wide")

@st.cache_resource
def load_system():
    init_models(use_nli=False)
    return load_model()

with st.spinner("Loading models..."):
    model = load_system()

def get_badge(label):
    if label == 'correct':
        return "🟢 Correct"
    elif label == 'contradictory':
        return "🟠 Contradictory"
    else:
        return "🔴 Incorrect"

def extract_qa_pairs(text):
    """Simple heuristic to extract numbered Q&A pairs from pasted text."""
    pairs = []
    current_q = ""
    current_a = []
    q_num = 1
    
    for line in text.split('\n'):
        match = re.match(r'^(\d+)[\.\)]\s*(.*)', line.strip())
        if match:
            if current_q:
                pairs.append({"#": q_num, "Question": current_q, "Answer": " ".join(current_a).strip()})
            q_num = int(match.group(1))
            current_q = match.group(2)
            current_a = []
        elif line.strip():
            current_a.append(line.strip())
            
    if current_q:
        pairs.append({"#": q_num, "Question": current_q, "Answer": " ".join(current_a).strip()})
        
    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["#", "Question", "Answer"])

def run_prediction(q_text, refs, student_ans):
    """Wrapper to handle caching and prediction."""
    q_hash = hashlib.md5("".join(refs).encode()).hexdigest()
    record = {
        'question_id': f'demo_q_{q_hash}',
        'question_text': q_text,
        'reference_answers': refs,
        'student_answer': student_ans,
        'domain': 'demo'
    }
    cache_reference_embeddings([record])
    pred_label, prob_vector = predict_single(model, record)
    evidence = generate_evidence(record, pred_label, prob_vector)
    return pred_label, evidence

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------
st.title("🎓 ED-05 Bias-Resistant Semantic Grader")

tab1, tab2 = st.tabs(["Single Answer", "Batch: Photographed Answer Sheet"])

# =============================================================================
# SCREEN 1: Single Answer
# =============================================================================
with tab1:
    st.header("Grade a Single Answer")
    
    q_input = st.text_input("Question", "Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.")
    
    st.subheader("Reference Answers")
    if 'ref_inputs' not in st.session_state:
        st.session_state.ref_inputs = ["Terminal 1 and the positive terminal are separated by the gap"]
        
    for i in range(len(st.session_state.ref_inputs)):
        st.session_state.ref_inputs[i] = st.text_input(f"Reference {i+1}", value=st.session_state.ref_inputs[i], key=f"s1_ref_{i}")
        
    if st.button("+ Add Reference"):
        st.session_state.ref_inputs.append("")
        st.rerun()
        
    s_input = st.text_area("Student's Answer", "Because there is a gap between terminal 1 and the positive terminal.")
    
    if st.button("Grade", type="primary"):
        refs = [r.strip() for r in st.session_state.ref_inputs if r.strip()]
        if not refs:
            st.error("Please provide at least one reference answer.")
        elif not s_input.strip():
            st.error("Please provide a student answer.")
        else:
            pred_label, evidence = run_prediction(q_input, refs, s_input)
            
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"### Result: {get_badge(pred_label)}")
            with col2:
                conf = evidence['confidence']
                conf_pct = conf['score'] * 100
                if conf['confident']:
                    st.success(f"Confidence: **{conf_pct:.1f}%**")
                else:
                    st.warning(f"Confidence: **{conf_pct:.1f}%** — ⚠️ Flagged for human review")
            
            st.markdown(f"**Evidence:** {evidence['evidence_string']}")
            if evidence['negation']['has_negation_cue']:
                st.caption(f"ℹ️ Note: {evidence['negation']['note']}")


# =============================================================================
# SCREEN 2: Batch Grading (Photographed Sheets)
# =============================================================================
with tab2:
    st.header("Batch: Grade a Photographed Answer Sheet")
    
    if "ref_df" not in st.session_state:
        st.session_state.ref_df = pd.DataFrame(columns=["#", "Question", "Answer"])
    if "stu_df" not in st.session_state:
        st.session_state.stu_df = pd.DataFrame(columns=["#", "Question", "Answer"])

    # --- Step 1: Reference Answers ---
    st.subheader("Step 1: Reference Answers")
    ref_mode = st.radio("Input method for References:", ["Upload Image(s)", "Paste Text (Numbered)"], horizontal=True)
    
    if ref_mode == "Paste Text (Numbered)":
        ref_text = st.text_area("Paste numbered questions and answers:", height=150,
                                placeholder="1. What is X?\nIt is Y.\n2. Why Z?\nBecause A.")
        if st.button("Extract References"):
            st.session_state.ref_df = extract_qa_pairs(ref_text)
            st.rerun()
    else:
        st.file_uploader("Upload Reference Answer Sheet (JPG/PNG)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="ref_uploader")
        if st.button("Run OCR on References"):
            # Mock OCR extraction
            st.session_state.ref_df = pd.DataFrame([
                {"#": 1, "Question": "What is the capital of France?", "Answer": "Paris"},
                {"#": 2, "Question": "Why does an open switch impact a circuit?", "Answer": "It creates a gap in the path."}
            ])
            st.rerun()

    if not st.session_state.ref_df.empty:
        st.markdown("**Confirm/Edit Reference Answers** (Errors here affect every student, so please review carefully)")
        st.session_state.ref_df = st.data_editor(st.session_state.ref_df, num_rows="dynamic", key="ref_editor", use_container_width=True)

    st.divider()

    # --- Step 2: Student Sheet ---
    st.subheader("Step 2: Student Sheet")
    stu_mode = st.radio("Input method for Student Answers:", ["Upload Image(s)", "Paste Text (Numbered)"], horizontal=True)
    
    if stu_mode == "Paste Text (Numbered)":
        stu_text = st.text_area("Paste student's numbered answers:", height=150,
                                placeholder="1. What is X?\nI think it is Y.\n2. Why Z?\nBecause B.")
        if st.button("Extract Student Answers"):
            st.session_state.stu_df = extract_qa_pairs(stu_text)
            st.rerun()
    else:
        st.file_uploader("Upload Student Answer Sheet (JPG/PNG)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="stu_uploader")
        if st.button("Run OCR on Student Sheet"):
            # Mock OCR extraction matching the mock questions
            st.session_state.stu_df = pd.DataFrame([
                {"#": 1, "Question": "What is the capital of France?", "Answer": "I think it is Paris."},
                {"#": 2, "Question": "Why does an open switch impact a circuit?", "Answer": "Because it doesn't create a gap."}
            ])
            st.rerun()

    if not st.session_state.stu_df.empty:
        st.markdown("**Confirm/Edit Student Answers** (Fix any OCR transcription errors here)")
        st.session_state.stu_df = st.data_editor(st.session_state.stu_df, num_rows="dynamic", key="stu_editor", use_container_width=True)
        
    st.divider()

    # --- Step 3: Grade All ---
    st.subheader("Step 3: Grade All")
    if st.button("Grade All", type="primary"):
        if st.session_state.ref_df.empty or st.session_state.stu_df.empty:
            st.error("Please extract and confirm both Reference and Student answers first.")
        else:
            results = []
            
            # Map references by question number
            ref_map = {}
            for _, row in st.session_state.ref_df.iterrows():
                q_num = row['#']
                if q_num not in ref_map:
                    ref_map[q_num] = {'q': row['Question'], 'refs': []}
                ref_map[q_num]['refs'].append(row['Answer'])
            
            for _, stu_row in st.session_state.stu_df.iterrows():
                q_num = stu_row['#']
                stu_ans = stu_row['Answer']
                
                if q_num in ref_map:
                    q_text = ref_map[q_num]['q']
                    refs = ref_map[q_num]['refs']
                    
                    pred_label, evidence = run_prediction(q_text, refs, stu_ans)
                    
                    results.append({
                        '#': q_num,
                        'Question': q_text,
                        'Student Answer (from OCR)': stu_ans,
                        'Label': get_badge(pred_label),
                        'Confidence': f"{evidence['confidence']['score']*100:.1f}%",
                        'Review?': '⚠️ Yes' if not evidence['confidence']['confident'] else 'No',
                        '_evidence': evidence['evidence_string'],
                        '_note': evidence['negation']['note'] if evidence['negation']['has_negation_cue'] else ""
                    })
                else:
                    results.append({
                        '#': q_num,
                        'Question': stu_row['Question'],
                        'Student Answer (from OCR)': stu_ans,
                        'Label': "N/A",
                        'Confidence': "N/A",
                        'Review?': "Missing Ref",
                        '_evidence': "",
                        '_note': ""
                    })
            
            if results:
                # Sort so 'Review?' == '⚠️ Yes' is at the top
                results.sort(key=lambda x: 0 if x['Review?'] == '⚠️ Yes' else 1)
                
                for r in results:
                    with st.expander(f"Q{r['#']}: {r['Label']} | {r['Review?']} | {r['Confidence']}"):
                        st.markdown(f"**Question:** {r['Question']}")
                        st.markdown(f"**Student Answer:** {r['Student Answer (from OCR)']}")
                        st.markdown(f"**Evidence:** {r['_evidence']}")
                        if r['_note']:
                            st.caption(f"ℹ️ Note: {r['_note']}")
