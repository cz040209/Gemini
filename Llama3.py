import os
import openai

# Fetch the API key from the environment variable (for Sambanova)
api_key = os.environ.get("10931f2d-bd6f-4ef0-adad-975aedc465ec")

# Check if the API key exists, otherwise exit with an error message
if not api_key:
    print("Error: API key is not set in the environment.")
    exit(1)

# Set the OpenAI API key for communication with OpenAI (Meta-Llama model)
openai.api_key = api_key
openai.api_base = "https://api.sambanova.ai/v1"

def chatbot():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    
    # Initialize conversation history
    conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Generate a response using OpenAI API with Meta-Llama 3.3-70B-Instruct model
            response = openai.ChatCompletion.create(
                model="Meta-Llama-3.3-70B-Instruct",  # Specify the Llama model here
                messages=conversation_history,
                temperature=0.7,  # Controls randomness in responses
                max_tokens=150,   # Max length of the response
                top_p=0.1         # You can set top_p as per your preferences
            )
            
            # Get the assistant's response
            bot_response = response['choices'][0]['message']['content']
            
            # Print the assistant's response
            print(f"Chatbot: {bot_response}")
            
            # Add assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": bot_response})
        
        except openai.error.OpenAIError as e:
            print(f"Error: {e}")
            print("There was an error communicating with the OpenAI API. Please try again.")
        
        except Exception as e:
            print(f"Unexpected Error: {e}")
            print("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    chatbot()
