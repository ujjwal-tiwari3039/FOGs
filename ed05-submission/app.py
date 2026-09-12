import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.features import init_models
from src.model import load_model, predict_single
from src.explain import generate_evidence
from src.metrics import CLASS_ORDER

st.set_page_config(page_title="ED-05 Grader", page_icon="🎓")

st.title("🎓 ED-05 Semantic Grader")
st.markdown("""
This demo evaluates short-answer responses against reference answers using
a calibrated sentence-embedding model. It resists keyword-stuffing and 
assesses its own confidence.
""")

@st.cache_resource
def load_system():
    init_models(use_nli=False)
    model = load_model()
    return model

with st.spinner("Loading models..."):
    model = load_system()

st.sidebar.header("Test an Answer")

question_text = st.sidebar.text_area("Question", "Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.")
reference_text = st.sidebar.text_area("Reference Answer(s) (one per line)", "Terminal 1 and the positive terminal are separated by the gap\nTerminal 1 and the positive terminal are not connected")
student_text = st.sidebar.text_area("Student Answer", "Because there is a gap between terminal 1 and the positive terminal.")

if st.sidebar.button("Grade Answer"):
    references = [r.strip() for r in reference_text.split('\n') if r.strip()]
    record = {
        'question_id': 'demo_q',
        'question_text': question_text,
        'reference_answers': references,
        'student_answer': student_text,
        'domain': 'demo'
    }
    
    pred_label, prob_vector = predict_single(model, record)
    evidence = generate_evidence(record, pred_label, prob_vector)
    
    st.header("Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Grade", pred_label.upper())
    with col2:
        conf = evidence['confidence']
        if conf['confident']:
            st.success("✅ High Confidence")
        else:
            st.warning(f"⚠️ Human Review Recommended\n\n{conf['reason']}")
            
    st.subheader("Probabilities")
    prob_df = pd.DataFrame({
        'Class': CLASS_ORDER,
        'Probability': prob_vector
    }).set_index('Class')
    st.bar_chart(prob_df)
    
    st.subheader("Explainability Evidence")
    st.write(evidence['evidence_string'])
    if evidence['negation']['has_negation_cue']:
        st.error(f"⚠ {evidence['negation']['note']}")
