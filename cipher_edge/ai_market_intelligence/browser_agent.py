import asyncio
from typing import Optional, Any, Dict

from cipher_edge.app_logger import get_logger
from cipher_edge.config.settings import settings

# This agent requires the `browser-use` library and a running Ollama instance.
# 1. pip install browser-use langchain-ollama
# 2. playwright install --with-deps chromium
# 3. ollama pull qwen2:7b (or your desired model)

logger = get_logger(__name__)

class BrowserAgent:
    """
    Uses the `browser-use` library with a local Ollama LLM to perform
    targeted research on a given cryptocurrency ticker.
    """
    def __init__(self, model_name: str = "qwen3:8b"):
        if not settings:
            logger.critical("Settings not loaded. BrowserAgent cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.llm_provider = settings.browser_agent_llm_provider
        self.llm: Optional[Any] = None
        self.agent_is_ready = False
        self.browser_use_agent_class: Optional[type] = None

        self.model_name = model_name

        try:
            self._initialize_llm()
            # Dynamically import browser_use only if LLM initialization is successful
            from browser_use import Agent as BrowserUseAgentLib
            self.browser_use_agent_class = BrowserUseAgentLib
            self.agent_is_ready = True
            logger.info("`browser-use` Agent component dynamically imported and ready.")
        except ImportError:
            logger.error("`browser-use` library not found. Please install it: pip install browser-use")
            logger.error("Also run: playwright install --with-deps chromium")
        except Exception as e:
            logger.error(f"BrowserAgent initialization failed: {e}", exc_info=True)


    def _initialize_llm(self):
        """Initializes the LLM object required by the browser-use library."""
        logger.info(f"Initializing LLM for BrowserAgent with provider: {self.llm_provider}")
        if self.llm_provider.lower() != "ollama":
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Please set to 'Ollama' in config.ini")
        
        try:
            from browser_use.llm import Ollama 
            
            self.llm = Ollama(model=self.model_name) 

            logger.info(f"BrowserAgent initialized with Ollama model: {self.model_name}")
        except ImportError:
            logger.error("A component of `browser-use` or `langchain-ollama` not found. Please ensure both are installed.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}", exc_info=True)
            logger.error("Ensure the Ollama application is running and the model is pulled.")
            raise

    async def _conduct_research(self, research_task: str, max_steps: int) -> Optional[Dict[str, Any]]:
        """
        Internal method to run the browser-use agent with a given task.
        """
        if not self.agent_is_ready or not self.llm or not self.browser_use_agent_class:
            logger.error("BrowserAgent is not ready. Cannot conduct research.")
            return None

        logger.info(f"Starting research task: '{research_task[:100]}...' (Max steps: {max_steps})")
        try:
            agent = self.browser_use_agent_class(
                llm=self.llm,
                task=research_task,
                use_vision=False,
                verbose=False
            )
            result = await agent.run(max_steps=max_steps)
            logger.info("Browser-use research task completed.")

            output_text = str(result.get('output', '')) if isinstance(result, dict) else str(result)
            logger.info(f"Browser Agent Final Output (first 500 chars): {output_text[:500]}...")
            return {"output": output_text, "full_result": result}

        except Exception as e:
            logger.error(f"An error occurred while running the browser-use agent: {e}", exc_info=True)
            return None

    async def research_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Creates and executes a research task for a specific cryptocurrency ticker.
        """
        task_prompt = (
            f"Research the cryptocurrency with the ticker '{ticker.upper()}'. "
            "Your goal is to find any significant news, analysis, or developments from the last 24-48 hours "
            "that could be affecting its price. "
            "Visit at least two reputable crypto news websites (like CoinDesk, Cointelegraph, Decrypt, The Block). "
            "Summarize your findings in a few paragraphs. Focus on tangible events like protocol upgrades, "
            "partnerships, regulatory news, security incidents, or major whale activity. "
            "Conclude your summary by listing the URLs of the most important articles you found."
        )
        
        max_steps = settings.browser_agent_max_steps if settings and settings.browser_agent_max_steps > 0 else 25
        return await self._conduct_research(research_task=task_prompt, max_steps=max_steps)

# Example of how to run the agent directly
async def main_browser_agent_example():
    """Main function to demonstrate BrowserAgent for a specific ticker."""
    try:
        browser_agent = BrowserAgent()
        if not browser_agent.agent_is_ready:
            logger.error("Browser agent could not be initialized. Exiting example.")
            return
    except Exception as e:
        logger.error(f"Failed to create BrowserAgent instance: {e}")
        return

    ticker_to_research = "SOL" # <-- Change this to research a different ticker
    
    logger.info(f"\n--- Starting Research for Ticker: {ticker_to_research.upper()} ---")
    
    result_data = await browser_agent.research_ticker(ticker_to_research)

    if result_data and result_data.get("output"):
        print("\n--- Research Result ---")
        print(result_data["output"])
        print("--- End of Result ---")
    else:
        logger.warning("Browser agent research failed or did not produce an output.")

if __name__ == "__main__":
    if settings and settings.browser_agent_enable:
        print("Running Browser Agent example...")
        print("Ensure the Ollama application is running in the background.")
        asyncio.run(main_browser_agent_example())
    else:
        print("BrowserAgent is not enabled in settings. Set BrowserAgent_Enable = True in your config.ini to run this example.")