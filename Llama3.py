import openai
import streamlit as st

# Set up the OpenAI API key using Streamlit secrets
api_key = st.secrets["SAMBANOVA"]["API_KEY"]

# Initialize OpenAI client with the API key and base URL
openai.api_key = api_key

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
        response = openai.ChatCompletion.create(
            model="Meta-Llama-3.3-70B-Instruct",  # Specify the model
            messages=st.session_state.conversation_history,  # Send the conversation history
            temperature=0.7,  # Control randomness of responses
            max_tokens=150,   # Limit the response length
            top_p=0.9,        # Control the diversity of responses
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Extract and return the model's response
        bot_response = response['choices'][0]['message']['content']
        return bot_response

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit input and output logic
user_input = st.text_input("You:", "")

if user_input:
    bot_response = get_bot_response(user_input)

    # Display the conversation
    st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})

    # Show the conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")

