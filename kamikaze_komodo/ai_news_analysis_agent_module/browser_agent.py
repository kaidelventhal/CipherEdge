# kamikaze_komodo/ai_news_analysis_agent_module/browser_agent.py
import asyncio
from typing import Optional, Any, Dict

from browser_use import Agent as BrowserUseAgent # Renamed to avoid conflict
from langchain_ollama import ChatOllama # Assuming this is the preferred LLM interface

from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BrowserAgent:
    """
    Uses browser-use with an LLM (e.g., Gemma 3 via Ollama) to perform targeted market research.
    This is an adaptation of the user-provided testBrowserUse.py script.
    """
    def __init__(self, llm_model_name: str = "gemma3:12b", ollama_base_url: Optional[str] = None, llm_context_window: int = 8192):
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.llm_context_window = llm_context_window # Default for many Gemma models
        self.llm: Optional[ChatOllama] = None
        
        try:
            llm_params: Dict[str, Any] = {
                "model": self.llm_model_name,
                "num_ctx": self.llm_context_window, # Context window size for Ollama
                "temperature": 0.1 # Low temperature for factual research
            }
            if self.ollama_base_url:
                llm_params["base_url"] = self.ollama_base_url
            
            self.llm = ChatOllama(**llm_params)
            logger.info(f"BrowserAgent initialized with LLM: {self.llm_model_name}, Context Window: {self.llm_context_window}")
        except Exception as e:
            logger.error(f"Error initializing Ollama LLM ({self.llm_model_name}) for BrowserAgent: {e}")
            logger.error("Please ensure Ollama is running and the model is pulled.")
            # self.llm will remain None, subsequent calls will fail gracefully.
            # Consider raising an exception if LLM is critical for instantiation.

    async def conduct_research(self, research_task: str, max_steps: int = 20, use_vision: bool = False) -> Optional[Dict[str, Any]]:
        """
        Conducts research based on the given task using browser-use.

        Args:
            research_task (str): The specific research task for the agent.
            max_steps (int): Maximum number of steps the browser agent can take.
            use_vision (bool): Whether to enable vision capabilities (if supported by LLM and browser-use version).

        Returns:
            Optional[Dict[str, Any]]: The result from the browser-use agent, typically includes 'output'.
                                      Returns None if LLM initialization failed or an error occurs.
        """
        if self.llm is None:
            logger.error("LLM not initialized. Cannot conduct research.")
            return None

        logger.info(f"Starting browser-use research task: '{research_task[:100]}...' (Max steps: {max_steps})")
        
        try:
            # Create the browser-use Agent instance
            agent = BrowserUseAgent(
                llm=self.llm,
                task=research_task,
                use_vision=use_vision, # Gemma 3 is primarily text, so vision might not be effective unless a multimodal version is used
                # verbose=True, # Enable for detailed browser-use logging
                # stop_sequences = ["Observation:", "Thought:"], # Example, may need tuning
            )

            # Run the agent
            # The result format can vary; often it's a dictionary containing 'output'
            # Or sometimes just a string.
            # Ensure browser-use is installed with `playwright install chromium --with-deps`
            result = await agent.run(max_steps=max_steps)
            
            logger.info("Browser-use research task completed.")
            if isinstance(result, dict):
                logger.debug(f"Browser agent raw result dictionary keys: {result.keys()}")
                # It seems 'output' or 'answer' is common for the final response.
                # If 'chat_history' is present, it can be very verbose.
                # We primarily care about the final textual output.
                final_output = result.get('output', result.get('answer', str(result)))
                logger.info(f"Browser Agent Output: {str(final_output)[:500]}...") # Log a snippet
                return {"output": final_output, "full_result": result} # Return a consistent structure
            elif isinstance(result, str):
                logger.info(f"Browser Agent Output (string): {result[:500]}...")
                return {"output": result, "full_result": result}
            else:
                logger.warning(f"Unexpected result type from browser-use agent: {type(result)}. Content: {str(result)[:500]}")
                return {"output": str(result), "full_result": result}


        except ImportError as ie:
            logger.error(f"ImportError with browser-use: {ie}. Is Playwright installed correctly with dependencies (playwright install --with-deps chromium)?")
            return None
        except Exception as e:
            logger.error(f"An error occurred while running the browser-use agent: {e}", exc_info=True)
            logger.error("Possible issues: Ollama server, network, task complexity, or website automation challenges.")
            return None

# Example Usage
async def main_browser_agent_example(settings_obj):
    if not settings_obj:
        logger.error("Settings not loaded for browser agent example.")
        return

    try:
        # Basic check if Ollama and model are available (optional pre-check)
        ollama_test_llm = ChatOllama(model=settings_obj.sentiment_llm_model, temperature=0.1)
        await ollama_test_llm.ainvoke("Hello") # Simple test
        logger.info(f"Ollama ({settings_obj.sentiment_llm_model}) seems accessible for BrowserAgent.")
        del ollama_test_llm
    except Exception as e:
        logger.error(f"Pre-check failed: Could not connect to Ollama or load {settings_obj.sentiment_llm_model}: {e}")
        return

    browser_agent = BrowserAgent(llm_model_name=settings_obj.sentiment_llm_model)
    if browser_agent.llm is None: # Check if LLM init failed in constructor
        return

    # Define a research task
    # Use specific keywords to guide the agent towards news.
    # Explicitly ask for summarization and sources.
    task = (
        "Investigate the recent news and general market sentiment for Ethereum (ETH) within the last 7-10 days. "
        "Browse 2-3 reputable cryptocurrency news websites (e.g., CoinTelegraph, Decrypt, The Block, but avoid CoinDesk for this test). "
        "Identify key news headlines, their sources, and briefly summarize the core message of each. "
        "Conclude with an overall sentiment assessment (e.g., bullish, bearish, neutral, mixed) based on the findings. "
        "Provide URLs for the articles found. "
        "Limit your Browse to a maximum of 4 steps per website visited. "
        "Focus on news related to price movements, technological updates (like upgrades), or significant partnerships."
    )

    result_data = await browser_agent.conduct_research(task, max_steps=25)

    if result_data and "output" in result_data:
        logger.info("\n--- Browser Agent Research Result ---")
        # The output from browser-use can be a long string containing thoughts, actions, observations.
        # We are interested in the final summarized answer.
        print(result_data["output"])
    elif result_data:
        logger.info("\n--- Browser Agent Raw Research Result ---")
        print(result_data)
    else:
        logger.warning("Browser agent research did not produce a result or failed.")

# if __name__ == "__main__":
#     # from kamikaze_komodo.config.settings import settings
#     # if settings:
#     #    # Ensure Playwright browsers are installed: playwright install chromium --with-deps
#     #    asyncio.run(main_browser_agent_example(settings))
#     # else:
#     #    print("Run this example from the project root or ensure settings are available.")
#     pass