import asyncio
from dotenv import load_dotenv
from browser_use import Agent
from langchain_ollama import ChatOllama

# Load environment variables (optional for local Ollama, but good practice)
load_dotenv()

async def research_bitcoin_sentiment():
    """
    Uses browser-use with Ollama (gemma3:12b) to research
    market sentiment and news about Bitcoin.
    """
    print("Initializing Ollama LLM (gemma3:12b)...")
    try:
        # Initialize the Ollama LLM
        # You can adjust num_ctx (context window size) if needed.
        # Larger values allow for more context but consume more resources.
        llm = ChatOllama(model="gemma3:12b", num_ctx=128000) # gemma3 models often have 8k context
        print("Ollama LLM initialized.")
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        print("Please ensure Ollama is running and the model 'gemma3:12b' is pulled.")
        return

    # Define the research task
    task = (
        "Research the current market sentiment surrounding Bitcoin. "
        "Find 2-3 recent news articles (from the last week if possible) that discuss Bitcoin's price, adoption, or regulatory news. "
        "Summarize the overall sentiment (e.g., bullish, bearish, neutral) and list the headlines and sources of the news articles found. "
        "Focus on reputable financial news sources or crypto-specific news sites."
        "Do not visit coindesk and do not stay on a website for more than 4 steps"
    )

    print(f"\nStarting browser-use agent with task: '{task}'")
    print("This might take a few minutes depending on the complexity and web page loading times...")

    try:
        # Create the browser-use Agent
        # use_vision=False is a good default if the model doesn't explicitly support it well
        # or if the task doesn't require image understanding.
        # For gemma3:12b, it's primarily text-based.
        agent = Agent(
            llm=llm,
            task=task,
            use_vision=False,
            # verbose=True, # For more detailed logging from browser-use
        )

        # Run the agent
        result = await agent.run(max_steps=50) 

        print("\n--- Research Complete ---")
        if isinstance(result, str):
            print(result)
        elif isinstance(result, dict) and 'output' in result:
            print(result['output'])
        else:
            print("Result from agent:")
            print(result)

    except Exception as e:
        print(f"\nAn error occurred while running the browser-use agent: {e}")
        print("Possible issues:")
        print("- Ollama server not responding or model not loaded correctly.")
        print("- Network connectivity problems for the browser.")
        print("- The task might be too complex or the websites visited might be problematic for automation.")

if __name__ == "__main__":
    # Check if Ollama is running and model is available (basic check)
    try:
        print("Checking Ollama status and model availability...")
        ollama_client_check = ChatOllama(model="gemma3:12b")
        ollama_client_check.invoke("Hello") # Simple test invocation
        print("Ollama and gemma3:12b seem to be accessible.")
        del ollama_client_check
    except Exception as e:
        print(f"Pre-check failed: Could not connect to Ollama or load gemma3:12b: {e}")
        print("Please ensure Ollama is running and you have pulled 'gemma3:12b' (ollama pull gemma3:12b).")
        exit(1)

    asyncio.run(research_bitcoin_sentiment())