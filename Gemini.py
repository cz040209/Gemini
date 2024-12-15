import os
import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import speech_recognition as sr  # For audio-to-text functionality
from PIL import Image
import pytesseract  # For OCR (Image to Text)

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chenz\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe'

# Your Hugging Face token
HF_TOKEN = "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"  # Replace with your token

# Load Summarization Model and Tokenizer (using Gemini for Summarization)
@st.cache_resource
def load_summarization_model():
    model_name = "google/gemini-summarization"  # Replace with the appropriate Gemini model name for summarization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize Translation Model (Gemini for English ↔ Chinese translation)
@st.cache_resource
def load_translation_model():
    model_name = "google/gemini-translation"  # Replace with the appropriate Gemini model name for translation
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Initialize models and tokenizers
summarization_tokenizer, summarization_model = load_summarization_model()
translation_model, translation_tokenizer = load_translation_model()

# Function to split text into manageable chunks for summarization
def split_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize text using Gemini model
def summarize_text(text):
    max_tokens = 1024  # Token limit for the model
    chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in chunks:
        inputs = summarization_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        summary_ids = summarization_model.generate(
            inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True
        )
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

# Function to translate text (English ↔ Chinese) using Gemini model
def translate_text(text, target_lang="zh"):
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translation_ids = translation_model.generate(
        inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True
    )
    translated_text = translation_tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Function for Audio-to-Text (Speech Recognition)
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Function for Image to Text (OCR)
def image_to_text(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Streamlit UI
def main():
    st.title("Text Processing with Gemini Models")
    
    # Upload PDF, Image, or Audio
    file_type = st.selectbox("Select the file type", ["PDF", "Audio", "Image"])
    
    if file_type == "PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
            st.write("Extracted text from PDF:")
            st.write(text)
            if st.button("Summarize Text"):
                summary = summarize_text(text)
                st.write("Summary:")
                st.write(summary)
            if st.button("Translate Text to Chinese"):
                translation = translate_text(text, target_lang="zh")
                st.write("Translated Text:")
                st.write(translation)
    
    elif file_type == "Audio":
        audio_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3", "m4a"])
        if audio_file:
            text = audio_to_text(audio_file)
            st.write("Extracted Text from Audio:")
            st.write(text)
            if st.button("Summarize Text"):
                summary = summarize_text(text)
                st.write("Summary:")
                st.write(summary)
            if st.button("Translate Text to Chinese"):
                translation = translate_text(text, target_lang="zh")
                st.write("Translated Text:")
                st.write(translation)
    
    elif file_type == "Image":
        image_file = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png"])
        if image_file:
            text = image_to_text(image_file)
            st.write("Extracted Text from Image:")
            st.write(text)
            if st.button("Summarize Text"):
                summary = summarize_text(text)
                st.write("Summary:")
                st.write(summary)
            if st.button("Translate Text to Chinese"):
                translation = translate_text(text, target_lang="zh")
                st.write("Translated Text:")
                st.write(translation)

if __name__ == "__main__":
    main()
