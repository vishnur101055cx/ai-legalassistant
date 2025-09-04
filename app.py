# app.py
import io, re, json
from collections import defaultdict
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

# If Tesseract not in PATH, set path here (uncomment and edit):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load keywords
with open("keywords.json","r",encoding="utf-8") as f:
    KEYWORDS = json.load(f)

# cached models
@st.cache_resource
def get_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error("Could not load summarizer model: " + str(e))
        return None

@st.cache_resource
def get_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

summarizer = get_summarizer()
embed_model = get_embed_model()

# helper functions
def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = []
        for p in reader.pages:
            try:
                text.append(p.extract_text() or "")
            except:
                pass
        return "\n".join(text)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    try:
        return data.decode("utf-8")
    except:
        return data.decode("latin-1", errors="ignore")

def extract_keywords(text: str):
    t = text.lower()
    info = {}
    def find_list(keys):
        seen=set(); found=[]
        for k in keys:
            kl=k.lower()
            if kl in t and kl not in seen:
                seen.add(kl); found.append(k)
        return found
    info["Court"] = find_list(KEYWORDS.get("courts",[]))
    info["Parties"] = find_list(KEYWORDS.get("parties",[]))
    info["Documents"] = find_list(KEYWORDS.get("documents",[]))
    info["Outcomes"] = find_list(KEYWORDS.get("outcomes",[]))
    dates=[]
    for pat in KEYWORDS.get("dates",[]):
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            if isinstance(m, tuple):
                dates.append(" ".join([x for x in m if x]))
            else:
                dates.append(m)
    info["Dates"]=dates
    return info

def simplify_text(text: str) -> str:
    s=text
    replacements = {
        "petitioner":"person who filed the case",
        "respondent":"other side",
        "impugned":"challenged",
        "hereby":"now",
        "forthwith":"immediately"
    }
    for k,v in replacements.items():
        s=s.replace(k,v)
        s=s.replace(k.capitalize(),v.capitalize())
    return s

def chunk_text(text, max_words=300):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_any_length(text):
    if not summarizer:
        return "Summarizer not available."
    try:
        lang = detect(text[:500])
    except:
        lang="en"
    working=text
    if lang!="en":
        try:
            working = GoogleTranslator(source="auto", target="en").translate(text)
        except:
            working=text
    if len(working.split()) < 60:
        summary_en = working
    else:
        parts = chunk_text(working, 280)
        partials=[]
        for p in parts:
            s = summarizer(p, max_length=160, min_length=60, do_sample=False)[0]["summary_text"]
            partials.append(s)
        combined=" ".join(partials)
        summary_en = summarizer(combined, max_length=180, min_length=70, do_sample=False)[0]["summary_text"]
    return simplify_text(summary_en)

def translate_summary(summary, code):
    if not summary: return ""
    try:
        return GoogleTranslator(source="auto", target=code).translate(summary)
    except:
        return "Translation unavailable."

# predictor module (uses predictor_similarity.py)
import predictor_similarity

def check_fake_or_real(doc_text:str, threshold:float=0.45):
    results = predictor_similarity.predict_similar_cases(doc_text, top_k=5)
    top_sim = results[0]["similarity"] if results else 0.0
    is_real = top_sim >= threshold
    return is_real, top_sim, results

def aggregate_outcome(results):
    scores=defaultdict(float)
    for r in results:
        scores[r['outcome']] += r['similarity']
    if not scores: return "No similar outcomes found"
    suggested = max(scores.items(), key=lambda x: x[1])[0]
    return suggested

# ---------- UI ----------
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ AI Legal Assistant — Camera / Upload / Paste")

tabs = st.tabs(["Camera (OCR)","Upload File","Paste Text","Predictor (Similarity)"])

# Camera tab
with tabs[0]:
    st.header("Camera OCR")
    st.write("Use your camera to capture the document. Then the app will OCR and process it.")
    img = st.camera_input("Take a photo of the legal document")
    if img is not None:
        image = Image.open(img)
        st.image(image, caption="Captured image")
        text = pytesseract.image_to_string(image)
        st.subheader("OCR Text (preview)")
        st.text_area("", value=text[:10000], height=220)
        if st.button("Process camera text"):
            with st.spinner("Processing..."):
                is_real, sim, results = check_fake_or_real(text)
                if is_real:
                    st.success(f"Document likely real (top similarity {sim:.3f})")
                    summary = summarize_any_length(text)
                    st.subheader("Simplified summary")
                    st.write(summary)
                    st.subheader("Key highlights")
                    st.json(extract_keywords(text))
                    lang = st.selectbox("Translate summary to", ["English","Malayalam","Hindi","Tamil"])
                    code_map = {"English":"en","Malayalam":"ml","Hindi":"hi","Tamil":"ta"}
                    if code_map[lang]!="en":
                        st.write(translate_summary(summary, code_map[lang]))
                    st.subheader("Top similar cases & outcomes")
                    for r in results:
                        st.markdown(f"**Case {r['case_id']} ({r['year']}) — sim {r['similarity']:.3f}**")
                        st.write(r['text'][:300]+"...")
                        st.write("Outcome: "+str(r['outcome']))
                    st.success("Suggested possible outcome: " + aggregate_outcome(results))
                else:
                    st.error(f"Document NOT similar to corpus (top sim {sim:.3f}). May be fake or out-of-domain.")

# Upload tab
with tabs[1]:
    st.header("Upload File")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf","docx","txt"])
    if uploaded is not None:
        text = read_file(uploaded)
        st.subheader("Extracted Text (preview)")
        st.text_area("", value=text[:10000], height=220)
        if st.button("Process uploaded file"):
            with st.spinner("Processing..."):
                is_real, sim, results = check_fake_or_real(text)
                if is_real:
                    st.success(f"Document likely real (top similarity {sim:.3f})")
                    summary = summarize_any_length(text)
                    st.subheader("Simplified summary")
                    st.write(summary)
                    st.subheader("Key highlights")
                    st.json(extract_keywords(text))
                    lang = st.selectbox("Translate summary to", ["English","Malayalam","Hindi","Tamil"])
                    code_map = {"English":"en","Malayalam":"ml","Hindi":"hi","Tamil":"ta"}
                    if code_map[lang]!="en":
                        st.write(translate_summary(summary, code_map[lang]))
                    st.subheader("Top similar cases & outcomes")
                    for r in results:
                        st.markdown(f"**Case {r['case_id']} ({r['year']}) — sim {r['similarity']:.3f}**")
                        st.write(r['text'][:300]+"...")
                        st.write("Outcome: "+str(r['outcome']))
                    st.success("Suggested possible outcome: " + aggregate_outcome(results))
                else:
                    st.error(f"Document NOT similar to corpus (top sim {sim:.3f}). May be fake or out-of-domain.")

# Paste text tab
with tabs[2]:
    st.header("Paste Text")
    paste_text = st.text_area("Paste legal text here", height=300)
    if st.button("Process pasted text"):
        if not paste_text.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Processing..."):
                is_real, sim, results = check_fake_or_real(paste_text)
                if is_real:
                    st.success(f"Document likely real (top similarity {sim:.3f})")
                    summary = summarize_any_length(paste_text)
                    st.subheader("Simplified summary")
                    st.write(summary)
                    st.subheader("Key highlights")
                    st.json(extract_keywords(paste_text))
                    lang = st.selectbox("Translate summary to", ["English","Malayalam","Hindi","Tamil"])
                    code_map = {"English":"en","Malayalam":"ml","Hindi":"hi","Tamil":"ta"}
                    if code_map[lang]!="en":
                        st.write(translate_summary(summary, code_map[lang]))
                    st.subheader("Top similar cases & outcomes")
                    for r in results:
                        st.markdown(f"**Case {r['case_id']} ({r['year']}) — sim {r['similarity']:.3f}**")
                        st.write(r['text'][:300]+"...")
                        st.write("Outcome: "+str(r['outcome']))
                    st.success("Suggested possible outcome: " + aggregate_outcome(results))
                else:
                    st.error(f"Document NOT similar to corpus (top sim {sim:.3f}). May be fake or out-of-domain.")

# Predictor tab
with tabs[3]:
    st.header("Predictor (Similarity)")
    doc_for_match = st.text_area("Paste the legal document you want to match:", height=300)
    if st.button("Find similar past cases and suggest outcome"):
        if not doc_for_match.strip():
            st.warning("Please paste the document text first.")
        else:
            with st.spinner("Searching for similar cases..."):
                results = predictor_similarity.predict_similar_cases(doc_for_match, top_k=5)
            st.subheader("Top similar past cases")
            for r in results:
                st.markdown(f"**Case {r['case_id']} ({r['year']}) — similarity: {r['similarity']:.3f}**")
                st.write(r['text'][:400] + ("…" if len(r['text'])>400 else ""))
                st.markdown(f"**Outcome:** {r['outcome']}")
                st.write("---")
            suggested = aggregate_outcome(results)
            st.success(f"Suggested possible outcome (weighted): **{suggested}**")
