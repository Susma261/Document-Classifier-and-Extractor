# app.py
import streamlit as st
from graph_pipeline import build_graph
from langchain.document_loaders import PyPDFLoader
from PIL import Image
import tempfile
import os
import json
import re
from datetime import datetime
import pytesseract
import cv2
import sqlite3
from dotenv import load_dotenv

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\SusmaRanganathan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Doc Classifier", layout="centered")
st.title(" Document Type Classifier and Extractor")

uploaded_file = st.file_uploader("Upload a PDF, Image (JPG, PNG)", type=["pdf", "png", "jpg", "jpeg"])

TESSERACT_CONFIG = "--oem 3 --psm 6"

def preprocess_image_for_ocr(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    processed_path = path.replace(".", "_processed.")
    cv2.imwrite(processed_path, thresh)
    return processed_path

def extract_json_string(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\{[\s\S]+?\})', text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return None

def log_to_db(file_name, timestamp, doc_type, extracted_data):
    conn = sqlite3.connect('db/logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                (file_name TEXT, timestamp TEXT, document_type TEXT, extracted_data TEXT)''')
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?)",
              (file_name, timestamp, doc_type, json.dumps(extracted_data)))
    conn.commit()
    conn.close()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    processed_path = None

    try:
        document_text = ""
        file_type = uploaded_file.name.lower()

        if file_type.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            full_text = " ".join([p.page_content for p in pages])
            document_text = full_text[:8000]

        elif file_type.endswith((".png", ".jpg", ".jpeg")):
            processed_path = preprocess_image_for_ocr(tmp_path)
            img = Image.open(processed_path)
            ocr_text = pytesseract.image_to_string(img, lang="eng", config=TESSERACT_CONFIG)
            document_text = ocr_text[:8000]

        else:
            st.error("Unsupported file type.")
            st.stop()

        graph = build_graph()
        result = graph.invoke({"doc": document_text})

        st.subheader(" Classification Result")
        st.success(f"**This document is classified as:** {result['label']}")

        st.subheader(" Extracted Information")
        extracted_data = extract_json_string(result["extracted_info"])
        if extracted_data:
            st.json(extracted_data)
        else:
            st.warning("Could not parse extracted information as JSON. Raw output:")
            st.text(result["extracted_info"])

        log_to_db(
            uploaded_file.name,
            datetime.now().isoformat(),
            result["label"],
            extracted_data if extracted_data else result["extracted_info"]
        )
        print("\n===== LOG OUTPUT TO TERMINAL =====")
        print("Filename:", uploaded_file.name)
        print("Classified As:", result["label"])
        print("Extracted Data:")
        print(json.dumps(extracted_data if extracted_data else result["extracted_info"], indent=4))


    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if processed_path and os.path.exists(processed_path):
            os.unlink(processed_path)
