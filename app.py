# app.py
import streamlit as st
import pdfplumber
import re
import io
import joblib
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Make sure NLTK resources exist
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Try to import transformers and keybert (optional, better results if available)
HAS_TRANSFORMERS = False
HAS_KEYBERT = False
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    HAS_KEYBERT = True
except Exception:
    HAS_KEYBERT = False

# Load your saved classifier and vectorizer (if present)
MODEL_PATH = "news_classifier.pkl"
VECT_PATH = "tfidf_vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    model = None
    vectorizer = None
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
    except Exception as e:
        st.warning(f"Could not load model/vectorizer from disk. Make sure '{MODEL_PATH}' and '{VECT_PATH}' exist. Error: {e}")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

# Optional: load summarizer pipeline (if transformers installed)
@st.cache_resource
def get_summarizer():
    if not HAS_TRANSFORMERS:
        return None
    # Use a fast summarization model
    try:
        pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
        return pipe
    except Exception:
        try:
            # fallback default
            pipe = pipeline("summarization")
            return pipe
        except Exception:
            return None

summarizer = get_summarizer()

# Optional: KeyBERT instance
@st.cache_resource
def get_keybert():
    if not HAS_KEYBERT:
        return None
    try:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        kw_model = KeyBERT(model=embed_model)
        return kw_model
    except Exception:
        return None

kw_model = get_keybert()

### Utility functions ###
def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text_parts.append(ptext)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    full_text = "\n\n".join(text_parts)
    return full_text

def clean_text(text: str) -> str:
    # Basic cleanup: remove multiple newlines, headers/footers heuristics
    text = re.sub(r'\n{2,}', '\n\n', text)
    # remove long sequences of whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # remove page numbers like "Page 1"
    text = re.sub(r'Page \d+','', text, flags=re.IGNORECASE)
    return text.strip()

def split_into_articles(text: str, min_chars=200) -> list:
    """
    Heuristic splitting:
     - split on double newlines (common paragraph breaks)
     - if parts are too short, group consecutive paragraphs until chunk ~min_chars
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    curr = ""
    for p in paras:
        if len(p) < 30 and curr:  # likely continuation or header
            curr += " " + p
            continue
        if not curr:
            curr = p
        else:
            if len(curr) >= min_chars:
                chunks.append(curr)
                curr = p
            else:
                curr += " " + p
    if curr:
        chunks.append(curr)
    # Filter out tiny chunks
    chunks = [c for c in chunks if len(c) > 100]
    return chunks

def get_headline_from_chunk(chunk: str) -> str:
    # Use first short-ish sentence as headline, else first sentence
    sents = sent_tokenize(chunk)
    for s in sents:
        words = s.split()
        if 3 < len(words) <= 18:  # plausible headline
            return s.strip()
    # fallback: first sentence trimmed
    return sents[0].strip() if sents else (chunk[:80] + "...")

def summarize_chunk(chunk: str, max_length=60) -> str:
    # Prefer transformer summarizer if available
    if summarizer:
        try:
            # split long texts into windows for pipeline if necessary
            # pipeline handles truncation but we'll do a simple approach
            summary = summarizer(chunk, max_length=max_length, min_length=20, do_sample=False)
            return summary[0]['summary_text'].strip()
        except Exception:
            pass
    # Fallback naive summarization: take the top sentences by TF-IDF score
    return naive_tfidf_summary(chunk, n_sentences=2)

def naive_tfidf_summary(chunk: str, n_sentences=2) -> str:
    sents = sent_tokenize(chunk)
    if len(sents) <= n_sentences:
        return chunk
    # build tfidf on the sentences
    vectorizer_local = TfidfVectorizer(stop_words='english', max_features=2000)
    try:
        X = vectorizer_local.fit_transform(sents)
        scores = np.asarray(X.sum(axis=1)).ravel()
        top_idx = np.argsort(scores)[-n_sentences:][::-1]
        top_sents = [sents[i] for i in sorted(top_idx)]
        return " ".join(top_sents)
    except Exception:
        # fallback to first N sentences
        return " ".join(sents[:n_sentences])

def extract_keywords(chunk: str, top_n=8) -> list:
    if kw_model:
        try:
            kws = kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)
            return [k[0] for k in kws]
        except Exception:
            pass
    # fallback with TF-IDF on chunk sentences
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        # vectorize the chunk as a single document, but we need feature names
        tfidf.fit([chunk])
        feat = np.array(tfidf.get_feature_names_out())
        vec = tfidf.transform([chunk]).toarray().ravel()
        if vec.sum() == 0:
            return []
        top_indices = vec.argsort()[-top_n:][::-1]
        return list(feat[top_indices])
    except Exception:
        # final fallback: pick most common words
        words = [w.lower() for w in re.findall(r'\w+', chunk) if w.lower() not in STOPWORDS and len(w) > 3]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [t[0] for t in top]

def classify_chunk(chunk: str):
    if (model is None) or (vectorizer is None):
        return {"label": "Unknown", "prob": None}
    try:
        v = vectorizer.transform([chunk])
        pred = model.predict(v)[0]
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(v)[0]
        return {"label": CATEGORIES[pred], "prob": probs}
    except Exception as e:
        return {"label": "Error", "prob": None, "error": str(e)}

def process_pdf_and_output(text):
    clean = clean_text(text)
    articles = split_into_articles(clean)
    results = []
    for i, art in enumerate(articles):
        headline = get_headline_from_chunk(art)
        summary = summarize_chunk(art)
        keywords = extract_keywords(art, top_n=8)
        classification = classify_chunk(art)
        probs = classification.get("prob")
        prob_map = {}
        if probs is not None:
            for cat, p in zip(CATEGORIES, probs):
                prob_map[cat] = float(p)
        results.append({
            "article_id": i+1,
            "headline": headline,
            "summary": summary,
            "keywords": ", ".join(keywords),
            "predicted_label": classification.get("label"),
            "probabilities": prob_map,
            "full_text": art[:4000]  # cap storing big text
        })
    return results

### Streamlit UI ###
st.set_page_config(page_title="Newspaper -> Headlines, Summaries & Classification", layout="wide")
st.title("📰 Newspaper Miner — Headlines, Summaries, Keywords & Classification")
st.markdown("Upload a newspaper PDF and get extracted headlines, short summaries, keywords and predicted categories (World / Sports / Business / Sci/Tech).")

col1, col2 = st.columns([2,1])

with col1:
    uploaded_file = st.file_uploader("Upload Newspaper PDF", type=["pdf"])
    min_chars = st.slider("Minimum chars per chunk (heuristic):", min_value=100, max_value=2000, value=350, step=50)
    do_advanced = st.checkbox("Use advanced summarization & keywords (transformers/keybert) if available", value=True)
    #st.markdown("**Note:** If transformers/keybert are not installed or your machine has limited memory, uncheck the above box to use lightweight TF-IDF fallbacks.")

#with col2:
    # st.write("Model status:")
    # st.write(f"- Classifier loaded: {'Yes' if model is not None else 'No'}")
    # st.write(f"- Vectorizer loaded: {'Yes' if vectorizer is not None else 'No'}")
    # st.write(f"- Transformers available: {'Yes' if HAS_TRANSFORMERS else 'No'}")
    # st.write(f"- KeyBERT available: {'Yes' if HAS_KEYBERT else 'No'}")
    # st.write("---")
    # st.write("Tips:")
    # st.write("- For best results, use reasonably clean PDF (text-based, not scanned images).")
    # st.write("- If PDF pages are scanned images, use OCR (Tesseract) first; not included here.")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if not text or len(text.strip()) < 100:
        st.error("No readable text found in PDF. Is it scanned image-based? Use OCR or upload a text PDF.")
    else:
        st.info("Processing PDF — this may take a few seconds (longer if using transformers).")
        # update global summarizer/kw_model usage based on user choice
        # (We won't reinitialize heavy models here; user can rerun with checkbox toggled before upload)
        results = process_pdf_and_output(text)
        df_rows = []
        for r in results:
            row = {
                "Article ID": r["article_id"],
                "Headline": r["headline"],
                "Summary": r["summary"],
                "Keywords": r["keywords"],
                "Predicted Label": r["predicted_label"]
            }
            # flatten probabilities as columns if exist
            if r["probabilities"]:
                for cat in CATEGORIES:
                    row[f"Prob_{cat}"] = r["probabilities"].get(cat, None)
            df_rows.append(row)
        df = pd.DataFrame(df_rows)

        st.success(f"Found {len(results)} article-like chunks. Displaying top results:")
        for i, r in enumerate(results):
            with st.expander(f"Article {r['article_id']}: {r['headline']} — Predicted: {r['predicted_label']}"):
                st.write("**Summary:**")
                st.write(r["summary"])
                st.write("**Keywords:**")
                st.write(r["keywords"])
                st.write("**Probabilities:**")
                if r["probabilities"]:
                    prob_df = pd.DataFrame(list(r["probabilities"].items()), columns=["Category", "Probability"])
                    st.table(prob_df)
                else:
                    st.write("No probability scores (model may not support predict_proba).")
                st.write("**Extract (first 1000 chars of article):**")
                st.write(r["full_text"][:1000])

        st.markdown("---")
        st.subheader("Results table")
        st.dataframe(df)

        # Allow user to download results as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV of Results", data=csv, file_name="newspaper_results.csv", mime="text/csv")
