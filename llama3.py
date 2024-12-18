import openai
import streamlit as st

# Set up Sambanova API credentials from Streamlit secrets
api_key = st.secrets["SAMBANOVA"]["API_KEY"]  # Ensure you have this in your .streamlit/secrets.toml
base_url = "https://api.sambanova.ai/v1"     # Sambanova endpoint

# Configure OpenAI client to use Sambanova API
openai.api_key = api_key
openai.api_base = base_url

# Streamlit UI Components
st.title("Meta-Llama Chatbot")
st.write("This chatbot is powered by Meta-Llama 3.3 70B-Instruct via Sambanova API.")

# Chat history in Streamlit session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

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

# User Input in Streamlit
user_input = st.text_input("You:", "")

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
