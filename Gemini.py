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

# Load Summarization and Q&A Model (Gemini model)
@st.cache_resource
def load_gemini_model():
    model_name = "google/gemini-1.5"  # Gemini model name (example, update if needed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)  # Updated argument
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)  # Updated argument
    return tokenizer, model

# Initialize models and tokenizers
gemini_tokenizer, gemini_model = load_gemini_model()

# Function to split text into manageable chunks for summarization or Q&A
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

# Function to summarize text
def summarize_text(text):
    max_tokens = 1024  # Token limit for the model
    chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in chunks:
        inputs = gemini_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        summary_ids = gemini_model.generate(
            inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True
        )
        summary = gemini_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

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

# History storage - will store interactions as tuples (user_input, response_output)
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for a more premium look
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #1c1f24;  /* Dark background */
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1v0m2ju {
            background-color: #282c34;  /* Slightly lighter background */
        }
        .css-13ya6yb {
            background-color: #61dafb;  /* Button color */
            border-radius: 5px;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .css-10trblm {
            font-size: 18px;
            font-weight: bold;
            color: #282c34;
        }
        .css-3t9iqy {
            color: #61dafb;
            font-size: 20px;
        }
        .botify-title {
            font-family: 'Arial', sans-serif;
            font-size: 48px;
            font-weight: bold;
            color: #61dafb;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Botify Title
st.markdown('<h1 class="botify-title">Botify</h1>', unsafe_allow_html=True)

# Option to choose between PDF upload, manual input, or translation
option = st.selectbox("Choose input method:", ("Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"))

context_text = ""

# Handling different options
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
            context_text = pdf_text

            # Summarize text
            st.subheader("Summarize the PDF Content")
            if st.button("Summarize PDF", use_container_width=True):
                with st.spinner("Summarizing text..."):
                    summary = summarize_text(pdf_text)
                st.success("Summary generated!")
                st.write(summary)
                st.session_state.history.append(("PDF Upload", summary))

            # Q&A Section
            st.subheader("Ask a Question about the PDF Content")
            question = st.text_input("Ask your question:")
            if question:
                with st.spinner("Generating answer..."):
                    answer = summarize_text(pdf_text)  # Use summarize function for basic Q&A or fine-tune for specific Q&A
                st.write(answer)
                st.session_state.history.append(("PDF Q&A", answer))
        else:
            st.error("Failed to extract text. Please check your PDF file.")

elif option == "Enter Text Manually":
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text

        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text", use_container_width=True):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Manual Text", summary))

        # Q&A Section
        st.subheader("Ask a Question about the Entered Text")
        question = st.text_input("Ask your question:")
        if question:
            with st.spinner("Generating answer..."):
                answer = summarize_text(manual_text)  # Use summarize function for basic Q&A or fine-tune for specific Q&A
            st.write(answer)
            st.session_state.history.append(("Manual Text Q&A", answer))

elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if audio_file:
        with st.spinner("Transcribing audio to text..."):
            try:
                transcription = audio_to_text(audio_file)
                st.success("Transcription successful!")
                st.write(transcription)
                st.session_state.history.append(("Audio Upload", transcription))

                # Q&A Section for audio transcription
                st.subheader("Ask a Question about the Audio Content")
                question = st.text_input("Ask your question:")
                if question:
                    with st.spinner("Generating answer..."):
                        answer = summarize_text(transcription)  # Use summarize function for basic Q&A or fine-tune for specific Q&A
                    st.write(answer)
                    st.session_state.history.append(("Audio Q&A", answer))

            except Exception as e:
                st.error(f"Error: {e}")

elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if image_file:
        with st.spinner("Extracting text from image..."):
            image_text = image_to_text(image_file)
            st.success("Text extracted from image!")
            st.write(image_text)
            st.session_state.history.append(("Image Upload", image_text))

            # Q&A Section for image text
            st.subheader("Ask a Question about the Image Content")
            question = st.text_input("Ask your question:")
            if question:
                with st.spinner("Generating answer..."):
                    answer = summarize_text(image_text)  # Use summarizeTo incorporate conversational AI and Q&A features using the Gemini model into your existing Streamlit app, we need to enhance the logic to handle both conversational dialogue and question-answering capabilities. Here's how you can add those functionalities to your application:


# Add function for conversational AI (using Gemini model)
def generate_conversation_response(user_input, history):
    # Prepare the conversation input by joining past history
    conversation_context = " ".join(history[-5:])  # Limit to the last 5 exchanges
    input_text = f"{conversation_context} User: {user_input} Bot:"
    
    # Tokenize and generate a response
    inputs = summarization_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    response_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True)
    response = summarization_tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # Update history with the new user input and the bot response
    history.append(f"User: {user_input}")
    history.append(f"Bot: {response}")
    
    return response

# Function for Q&A using the model
def generate_qa_response(question, context):
    # Prepare the input for the question-answering task
    input_text = f"Context: {context} Question: {question} Answer:"
    
    # Tokenize and generate an answer
    inputs = summarization_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    answer_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True)
    answer = summarization_tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    
    return answer

# Handling the interaction type
interaction_type = st.radio("Select Interaction Type:", ["Conversation", "Q&A"])

if interaction_type == "Conversation":
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        with st.spinner("Generating response..."):
            # Handle conversation logic
            conversation_response = generate_conversation_response(user_input, st.session_state.history)
            st.write(f"Bot: {conversation_response}")
            st.session_state.history.append(f"User: {user_input}")
            st.session_state.history.append(f"Bot: {conversation_response}")
            
elif interaction_type == "Q&A":
    question = st.text_input("Ask a Question:", key="question")
    context_text = st.text_area("Provide Context (optional):")
    
    if question:
        with st.spinner("Generating answer..."):
            # Handle Q&A logic
            if context_text.strip():
                answer = generate_qa_response(question, context_text)
            else:
                answer = "Please provide some context for a meaningful answer."
            st.write(f"Answer: {answer}")

# Update the sidebar to show history of the current session
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, bot_response) in enumerate(zip(st.session_state.history[::2], st.session_state.history[1::2])):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Bot Response:** {bot_response}")
else:
    st.sidebar.write("No history yet.")
