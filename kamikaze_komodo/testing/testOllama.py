import ollama

def chat_with_ollama():
    """
    Allows the user to interact with a specified Ollama model.
    """
    # --- Configuration ---
    # You can change the model_name to any model you have available in Ollama.
    # Run 'ollama list' in your terminal to see available models.
    model_name = 'gemma3:12b' # Example: replace if needed

    print(f"Starting chat with Ollama model: {model_name}")
    print("Type 'quit', 'exit', or 'bye' to end the chat.")
    print("-" * 30)

    # --- Initialize conversation history ---
    # Ollama's chat endpoint can maintain context if you pass the history.
    messages = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Exiting chat. Goodbye! 👋")
                break

            # Add user's message to history
            messages.append({'role': 'user', 'content': user_input})

            # --- Send prompt to Ollama and stream response ---
            # stream=True provides a more interactive experience as tokens arrive.
            # stream=False will wait for the full response.
            response_stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True
            )

            print(f"AI ({model_name}): ", end="", flush=True)
            assistant_response = ""
            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    print(token, end="", flush=True)
                    assistant_response += token
                # Check for the 'done' status if not using stream=True,
                # or to handle the end of a streamed response.
                if chunk.get('done'):
                    print() # Newline after the full response
                    # Add assistant's full response to history
                    messages.append({'role': 'assistant', 'content': assistant_response})
                    break
            
            if not assistant_response: # If stream ended without content (e.g., error)
                print("\nNo response from model or stream ended.")
                # Optionally remove the last user message if there was no valid response
                if messages and messages[-1]['role'] == 'user':
                    messages.pop()


        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # You might want to clear messages or handle the error more gracefully
            break

if __name__ == '__main__':
    # --- Check if Ollama is running and the model is available ---
    try:
        client = ollama.Client()
        client.list() # Simple check to see if server is reachable
        print("Successfully connected to Ollama.")
        
        # You can add a check here to see if model_name exists if desired,
        # though ollama.chat will also error out if it doesn't.
        
    except Exception as e:
        print(f"Error connecting to Ollama or Ollama server not running: {e}")
        print("Please ensure the Ollama application is running.")
        exit()
        
    chat_with_ollama()