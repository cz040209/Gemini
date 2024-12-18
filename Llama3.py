import openai

# Set your API key
api_key = "10931f2d-bd6f-4ef0-adad-975aedc465ec"
openai.api_key = api_key

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
