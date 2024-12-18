import openai
import streamlit as st

# Set up the OpenAI API key and base URL using Streamlit secrets
api_key = st.secrets["SAMBANOVA"]["API_KEY"]
base_url = "https://api.sambanova.ai/v1"

# Initialize OpenAI client for Sambanova API
client = openai.OpenAI(api_key=api_key, base_url=base_url)

# Streamlit UI components
st.title("Meta-Llama Chatbot")
st.write("This is a chatbot powered by Meta-Llama 3.3 70B-Instruct.")

# Chat history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to get a response from the LLM model
def get_bot_response(user_input):
    # Append the user's input to the conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        # Request response from the Meta-Llama model via Sambanova API
        response = client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=st.session_state.conversation_history,
            temperature=0.7,
            max_tokens=150,
            top_p=0.9
        )

        # Extract and return the model's response
        bot_response = response.choices[0].message.content
        return bot_response

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit input and output logic
user_input = st.text_input("You:", "")

if user_input:
    bot_response = get_bot_response(user_input)

    # Display the conversation
    st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})

    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"Bot: {message['content']}")
