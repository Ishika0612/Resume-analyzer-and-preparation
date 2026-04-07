import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import json, tempfile, re
from io import BytesIO

load_dotenv()

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ---------- JSON SAFE PARSER ----------
def extract_json(text):
    try:
        return json.loads(text)
    except:
        pass
    try:
        text = text.strip("```json").strip("```")
        return json.loads(text)
    except:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

# ---------- PDF GENERATOR ----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(score, ats, q_data):
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Interview Preparation Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"ATS Score: {score}", styles['Heading2']))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Suitable Roles:", styles['Heading2']))
    for r in ats["Suitable_Roles"]:
        content.append(Paragraph("• " + r, styles['Normal']))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Missing Skills:", styles['Heading2']))
    for s in ats["Missing_Skills"]:
        content.append(Paragraph("• " + s, styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Interview Questions", styles['Heading2']))
    content.append(Spacer(1, 10))

    for section, qs in q_data.items():
        content.append(Paragraph(section, styles['Heading3']))
        for q in qs:
            content.append(Paragraph("• " + q, styles['Normal']))
            content.append(Spacer(1, 5))

    doc = SimpleDocTemplate(buffer)
    doc.build(content)

    buffer.seek(0)
    return buffer

# ---------- UI ----------
st.title("🚀 AI Resume Analyzer & Interview Prep")

uploaded_file = st.file_uploader("📄 Upload Resume", type=["pdf"])

# Disable button if no file
analyze_btn = st.button(
    "⚡ Analyze Resume",
    disabled=(uploaded_file is None)
)

# Warning if no file
if uploaded_file is None:
    st.info("📌 Please upload a resume to start analysis")
    st.stop()

# Save temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    file_path = tmp.name

st.success("✅ Resume Uploaded")

# ---------- MAIN LOGIC ----------
if analyze_btn:

    with st.spinner("Analyzing..."):

        # LOAD PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(texts, embedding_model)

        results = vectorstore.similarity_search(
            "skills, experience, projects, achievements", k=3
        )

        context = "\n".join([doc.page_content for doc in results])

        model = ChatMistralAI(model="mistral-small")

        # ---------- VALIDATION ----------
        keywords = ["education", "skills", "experience", "projects"]

        if not any(word in context.lower() for word in keywords):
            st.error("❌ This does not look like a resume.")
            st.stop()

        validation_prompt = f"""
Check if the following document is a resume.

Text:
{context}

Return ONLY YES or NO.
"""

        val_response = model.invoke(validation_prompt).content.strip()

        if "NO" in val_response.upper():
            st.error("❌ Invalid document. Please upload a resume.")
            st.stop()

        st.success("✅ Valid Resume Detected")

        # ---------- ATS ----------
        messages = [
            SystemMessage(content="Return ONLY valid JSON."),
            HumanMessage(content=f"""
Analyze this resume:

{context}

Return:
{{
"ATS_Score": number ,
"Suitable_Roles": [],
"Missing_Skills": []
}}
""")
        ]

        ats_raw = model.invoke(messages).content
        ats = extract_json(ats_raw)

        if ats is None:
            st.error("❌ Failed to analyze resume")
            st.stop()

        score = ats["ATS_Score"]

        # ---------- UI ----------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 ATS Score")
            st.progress(score / 100)
            st.write(score)

        with col2:
            st.subheader("💼 Suitable Roles")
            for r in ats["Suitable_Roles"]:
                st.write("✔️", r)

        st.subheader("⚠️ Missing Skills")
        for s in ats["Missing_Skills"]:
            st.write("❌", s)

    

        # ---------- QUESTIONS ----------
        q_prompt = f"""
Generate interview questions from this resume:

{context}

Return ONLY JSON:
{{
"HR": [],
"Technical": [],
"Project": []
}}
"""

        q_raw = model.invoke(q_prompt).content
        q_data = extract_json(q_raw)

        if q_data is None:
            st.error("❌ Failed to generate questions")
            st.stop()

        # ---------- PDF ----------
        pdf_file = create_pdf(score, ats, q_data)

        st.success("✅ PDF Ready!")

        st.download_button(
            "📥 Download Full Report (PDF)",
            pdf_file,
            file_name="AI_Resume_Report.pdf",
            mime="application/pdf"
        )