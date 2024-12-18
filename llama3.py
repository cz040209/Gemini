import openai
import streamlit as st
import PyPDF2
import io

# Set up Sambanova API credentials from Streamlit secrets
api_key = st.secrets["SAMBANOVA"]["API_KEY"]  # Ensure you have this in your .streamlit/secrets.toml
base_url = "https://api.sambanova.ai/v1"     # Sambanova endpoint

# Configure OpenAI client to use Sambanova API
openai.api_key = api_key
openai.api_base = base_url

# Streamlit UI Components
st.title("Meta-Llama Chatbot with PDF Q&A")
st.write("This chatbot is powered by Meta-Llama 3.3 70B-Instruct via Sambanova API.")

# Chat history in Streamlit session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Function to get a response from Sambanova API
def get_bot_response(user_input):
    # Append the user's input to the conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        # Call the Sambanova API via OpenAI client
        response = openai.ChatCompletion.create(
            model="Meta-Llama-3.3-70B-Instruct",  # Specify the Meta-Llama model
            messages=st.session_state.conversation_history,  # Include conversation history
            temperature=0.7,  # Control randomness
            max_tokens=150,   # Response length limit
            top_p=0.9         # Control diversity
        )

        # Extract and return the bot's response
        bot_response = response['choices'][0]['message']['content']
        return bot_response

    except Exception as e:
        # Handle API errors
        return f"An error occurred: {e}"

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    try:
        # Read the uploaded PDF file
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            pdf_text += page.extract_text()  # Extract text from each page
        return pdf_text
    except Exception as e:
        return f"Error reading PDF: {e}"

# File Uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file to analyze its content", type="pdf")

if uploaded_file:
    # Extract text from PDF and store it in session state
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("PDF content has been uploaded and processed successfully!")
    st.text_area("Extracted PDF Text", st.session_state.pdf_text, height=300)

    # Append the extracted PDF text to the system prompt
    st.session_state.conversation_history.append(
        {"role": "system", "content": f"The following is the extracted text from the uploaded PDF:\n\n{st.session_state.pdf_text}"}
    )

# User Input in Streamlit
user_input = st.text_input("Ask a question about the PDF or chat with the bot:", "")

if user_input:
    # Get the bot's response
    bot_response = get_bot_response(user_input)

    # Append the bot's response to the conversation history
    st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})

    # Display the conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"**You**: {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Bot**: {message['content']}")
