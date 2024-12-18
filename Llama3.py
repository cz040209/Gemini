import openai
import streamlit as st

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["SAMBANOVA"]["API_KEY"]

# Set up the OpenAI client with the API key and base URL
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.sambanova.ai/v1"
)

response = client.chat.completions.create(
    model="Meta-Llama-3.3-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.1,
    top_p=0.1
)

# Print the response from the API
print(response.choices[0].message.content)
