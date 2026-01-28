import streamlit as st
import whisper
import torch
import spacy
import os
import time
import yt_dlp
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. Page Config & Professional Styling ---
st.set_page_config(page_title="LectureAI | Professional", layout="wide")

st.markdown("""
    <style>
    /* Arctic Light Theme Colors */
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    
    /* Card-style Containers */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Header Styling */
    .main-header { font-size: 32px; font-weight: 800; color: #0F172A; }
    .input-label { font-weight: 700; color: #475569; margin-bottom: 8px; display: block; }

    /* Modern Indigo Button */
    .stButton button {
        background: #4F46E5;
        color: white !important;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
        height: 3rem;
        border: none;
    }
    .stButton button:hover { background: #4338CA; }

    /* Analytics Metrics */
    div[data-testid="stMetric"] {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #4F46E5 !important; border-bottom: 2px solid #4F46E5 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI Logic Functions ---
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return whisper_model, (model, tokenizer), nlp

whisper_model, bart_bundle, nlp = load_models()

def download_yt_audio(url):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'yt_audio.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.mp3"

def get_analytics(text):
    words = re.findall(r'\w+', text)
    count = len(words)
    reading_time = max(1, round(count / 210)) # Avg 210 WPM
    return count, reading_time

def extract_concepts(text):
    doc = nlp(text)
    concepts = {}
    for chunk in doc.noun_chunks:
        term = chunk.root.text.capitalize()
        if len(term) > 4 and term not in concepts:
            for sent in doc.sents:
                if chunk.text in sent.text:
                    concepts[term] = sent.text
                    break
        if len(concepts) >= 6: break
    return concepts

# --- 3. Sidebar Help Section ---
with st.sidebar:
    st.markdown("<h2 style='color: #4F46E5;'>LectureAI Help</h2>", unsafe_allow_html=True)
    with st.expander("YouTube Errors?"):
        st.write("1. Ensure the video is Public.")
        st.write("2. Age-restricted videos may fail.")
        st.write("3. Check your internet connection.")
    st.divider()
    st.caption("v2.6 Arctic Professional Edition")

# --- 4. Main UI Layout ---
st.markdown("<div class='main-header'>Lecture Intelligence Dashboard</div>", unsafe_allow_html=True)
st.write("Transform complex audio into actionable study notes and insights.")

st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<span class='input-label'>Upload Lecture File</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["mp3", "wav"], label_visibility="collapsed")
    st.caption("Accepted: MP3, WAV")

with col2:
    st.markdown("<span class='input-label'>YouTube Link</span>", unsafe_allow_html=True)
    yt_url = st.text_input("URL", placeholder="Paste YouTube link here...", label_visibility="collapsed")
    st.caption("Format: https://youtube.com/...")

st.markdown("<br>", unsafe_allow_html=True)
process_btn = st.button("Generate Intelligence Report")
st.markdown("</div>", unsafe_allow_html=True)

if process_btn:
    source_path = None
    if uploaded_file:
        source_path = "temp_local.mp3"
        with open(source_path, "wb") as f: f.write(uploaded_file.getbuffer())
    elif yt_url:
        with st.spinner("Downloading YouTube Content..."): source_path = download_yt_audio(yt_url)
    
    if source_path:
        try:
            with st.status("Neural Pipeline Processing...", expanded=True) as status:
                st.write("Transcribing Speech Patterns...")
                raw_text = whisper_model.transcribe(source_path)["text"]
                
                st.write("Generating Executive Summary...")
                model, tokenizer = bart_bundle
                inputs = tokenizer([raw_text[:1024]], return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                st.write("Performing Semantic Analysis...")
                concepts = extract_concepts(raw_text)
                word_count, read_time = get_analytics(raw_text)
                status.update(label="Analysis Complete", state="complete", expanded=False)

            # --- Results ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Word Count", word_count)
            m2.metric("Reading Time", f"{read_time} min")
            m3.metric("Key Terms", len(concepts))
            m4.metric("Privacy", "Offline")

            t1, t2, t3 = st.tabs(["EXECUTIVE SUMMARY", "CONCEPT GLOSSARY", "FULL TRANSCRIPT"])
            
            with t1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(summary)
                st.download_button("Export Summary", summary, "summary.txt")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with t2:
                for term, definition in concepts.items():
                    with st.expander(f"Concept: {term}"):
                        st.write(definition)
                        
            with t3:
                st.text_area("Transcription Text", raw_text, height=300)

        except Exception as e:
            st.error(f"Operational Error: {e}")
        finally:
            if source_path and os.path.exists(source_path): os.remove(source_path)
    else:
        st.error("Error: No data source detected. Please upload a file or paste a link.")