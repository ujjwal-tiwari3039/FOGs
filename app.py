import streamlit as st
import yaml
import joblib
import numpy as np
import pandas as pd
from src.embeddings import EmbeddingModel
from src.features import FeatureExtractor
from src.model import ShortAnswerClassifier
from src.counterfactuals import CounterfactualGenerator
from src.explainability import Explainer

st.set_page_config(page_title="Bias-Resistant Assessment", layout="wide")

@st.cache_resource
def load_resources():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    clf = ShortAnswerClassifier()
    clf.load(config['paths']['model_save_path'])

    metadata = joblib.load("results/metadata.joblib")
    embedder = EmbeddingModel(model_name=metadata['embedding_model'])
    extractor = FeatureExtractor(**metadata['feature_extractor_params'])

    # For CF4, we need a corpus. We'll load the processed train.csv
    train_df = pd.read_csv("data/processed/train.csv")
    cf_gen = CounterfactualGenerator(corpus=train_df['student'].tolist())

    explainer = Explainer(clf, extractor)

    return clf, embedder, extractor, cf_gen, explainer, metadata

def main():
    st.title("📝 Bias-Resistant Short-Answer Assessment")
    st.markdown("""
    This system assesses student answers based on **semantic meaning** rather than superficial keywords.
    It is designed to be robust against style changes and keyword repetition.
    """)

    try:
        clf, embedder, extractor, cf_gen, explainer, metadata = load_resources()
    except Exception as e:
        st.error(f"Failed to load model resources. Please run training first. Error: {e}")
        return

    col1, col2 = st.columns(2)

    with col1:
        ref_text = st.text_area("Reference Answer", "Terminal 1 and the positive terminal are separated by the gap")
        student_text = st.text_area("Student Answer", "The positive battery terminal is separated by a gap from terminal 1")

        assess_btn = st.button("Assess Answer")

    if assess_btn:
        with col2:
            # Process
            s_emb = embedder.encode([student_text])[0]
            r_emb = embedder.encode([ref_text])[0]

            feats = extractor.extract(
                s_emb.reshape(1, -1),
                r_emb.reshape(1, -1),
                [student_text],
                [ref_text]
            )

            result = explainer.explain(student_text, ref_text, s_emb, r_emb)

            # Display Prediction
            color = "green" if result['prediction'] == "CORRECT" else "red" if result['prediction'] == "CONTRADICTORY" else "orange"
            st.markdown(f"### Prediction: <span style='color:{color}'>{result['prediction']}</span>", unsafe_allow_html=True)

            # Probabilities
            st.write("**Class Probabilities:**")
            for label, prob in result['probabilities'].items():
                st.write(f"{label.capitalize()}: {prob:.4f}")

            st.metric("Semantic Similarity", f"{result['similarity']:.4f}")

            st.write("**Explanation:**")
            for point in result['evidence']:
                st.write(f"- {point}")

            # Counterfactuals
            st.divider()
            st.subheader("Robustness Check (Counterfactuals)")
            cfs = cf_gen.generate(student_text)

            cf_data = []
            for cf_key in ['cf1', 'cf2', 'cf3', 'cf4']:
                text = cfs[cf_key]
                # Predict for CF
                cf_s_emb = embedder.encode([text])[0]
                cf_feats = extractor.extract(
                    cf_s_emb.reshape(1, -1),
                    r_emb.reshape(1, -1),
                    [text],
                    [ref_text]
                )
                cf_prob = clf.predict_proba(cf_feats)[0]
                cf_label = metadata['label_map_inv'][np.argmax(cf_prob)] if 'label_map_inv' in metadata else "Unknown"
                # Since we didn't save label_map_inv, we can reconstruct it
                labels = ['correct', 'contradictory', 'incorrect']
                cf_label = labels[np.argmax(cf_prob)]

                cf_data.append({
                    "Variant": cf_key.upper(),
                    "Text": text,
                    "Prediction": cf_label.upper(),
                    "Prob(Correct)": f"{cf_prob[0]:.4f}"
                })

            st.table(pd.DataFrame(cf_data))

if __name__ == "__main__":
    main()
