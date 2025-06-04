# kamikaze_komodo/ai_news_analysis_agent_module/browser_agent.py
import asyncio
from typing import Optional, Any, Dict
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings # Import global settings

logger = get_logger(__name__)

class BrowserAgent:
    """
    Uses browser-use with a configured LLM (e.g., Vertex AI's Gemini)
    to perform targeted market research.
    """
    def __init__(self):
        if not settings:
            logger.critical("Settings not loaded. BrowserAgent cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.llm_provider = settings.browser_agent_llm_provider
        self.llm: Optional[Any] = None # To be initialized
        self.agent_is_ready = False

        try:
            self._initialize_llm()
            # Dynamically import browser_use only if LLM init is successful
            global BrowserUseAgent # Make it global for the method if loaded
            from browser_use import Agent as BrowserUseAgent
            self.agent_is_ready = True
            logger.info("browser-use Agent component dynamically imported.")
        except ImportError:
            logger.error("browser-use library not found. Please install it: pip install browser-use")
            logger.error("Also run: playwright install --with-deps chromium")
        except Exception as e:
            logger.error(f"BrowserAgent initialization failed: {e}")


    def _initialize_llm(self):
        logger.info(f"Initializing LLM for BrowserAgent with provider: {self.llm_provider}")
        if self.llm_provider == "VertexAI":
            if not settings.vertex_ai_project_id or not settings.vertex_ai_location:
                logger.error("Vertex AI project ID or location is not configured. BrowserAgent LLM will not work.")
                raise ValueError("Vertex AI project ID or location missing for BrowserAgent.")
            try:
                from langchain_google_vertexai import ChatVertexAI
                self.llm = ChatVertexAI(
                    project=settings.vertex_ai_project_id,
                    location=settings.vertex_ai_location,
                    model_name=settings.vertex_ai_browser_agent_model_name, # Use specific model for browser agent
                    temperature=0.2, # Slightly higher temp for research/summarization
                    # max_output_tokens=2048, # Optional
                )
                logger.info(f"BrowserAgent initialized with Vertex AI model: {settings.vertex_ai_browser_agent_model_name}")
                logger.info("Ensure Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS) are set for Vertex AI.")
            except ImportError:
                logger.error("langchain-google-vertexai not found. pip install langchain-google-vertexai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI LLM for BrowserAgent: {e}", exc_info=True)
                raise

        else:
            logger.error(f"Unsupported LLM provider for BrowserAgent: {self.llm_provider}")
            raise ValueError(f"Unsupported LLM provider for BrowserAgent: {self.llm_provider}")
        
        if self.llm is None:
            raise ValueError("BrowserAgent LLM initialization failed.")


    async def conduct_research(self, research_task: str, max_steps: int = 20, use_vision: bool = False) -> Optional[Dict[str, Any]]:
        """
        Conducts research based on the given task using browser-use.
        """
        if not self.agent_is_ready or self.llm is None:
            logger.error("BrowserAgent or its LLM not initialized. Cannot conduct research.")
            return None

        logger.info(f"Starting browser-use research task: '{research_task[:100]}...' (Max steps: {max_steps})")
        try:
            agent = BrowserUseAgent( # This is the dynamically imported class
                llm=self.llm,
                task=research_task,
                use_vision=use_vision,
                # verbose=True,
            )
            result = await agent.run(max_steps=max_steps)
            logger.info("Browser-use research task completed.")

            final_output_text = ""
            if isinstance(result, dict):
                logger.debug(f"Browser agent raw result dictionary keys: {result.keys()}")
                final_output_text = result.get('output', result.get('answer', str(result)))
            elif isinstance(result, str):
                final_output_text = result
            else:
                final_output_text = str(result)
                logger.warning(f"Unexpected result type from browser-use agent: {type(result)}. Content: {final_output_text[:500]}")

            logger.info(f"Browser Agent Output: {final_output_text[:500]}...")
            return {"output": final_output_text, "full_result": result}

        except Exception as e:
            logger.error(f"An error occurred while running the browser-use agent: {e}", exc_info=True)
            logger.error("Possible issues: LLM server, network, task complexity, or website automation challenges.")
            return None

async def main_browser_agent_example():
    if not settings or not settings.browser_agent_enable:
        logger.info("BrowserAgent is not enabled in settings or settings not loaded.")
        return
    if not settings.vertex_ai_project_id and settings.browser_agent_llm_provider == "VertexAI":
        logger.error("Vertex AI Project ID not set for BrowserAgent. Set GOOGLE_APPLICATION_CREDENTIALS.")
        return

    try:
        browser_agent = BrowserAgent()
        if not browser_agent.agent_is_ready:
            logger.error("Browser agent could not be initialized. Exiting example.")
            return
    except Exception as e:
        logger.error(f"Failed to create BrowserAgent: {e}")
        return

    task = (
        "What is the latest news regarding Ethereum's price action and upcoming upgrades in June 2025? "
        "Visit 2 reputable crypto news websites (e.g., Decrypt, Cointelegraph, but NOT CoinDesk). "
        "Summarize findings and list article URLs. Limit Browse to 4 steps per site."
    )
    result_data = await browser_agent.conduct_research(task, max_steps=settings.browser_agent_max_steps or 25)

    if result_data and "output" in result_data:
        logger.info("\n--- Browser Agent Research Result ---")
        print(result_data["output"])
    elif result_data:
        logger.info("\n--- Browser Agent Raw Research Result ---")
        print(result_data)
    else:
        logger.warning("Browser agent research did not produce a result or failed.")

if __name__ == "__main__":
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set for Vertex AI
    # Ensure Playwright browsers are installed: playwright install --with-deps chromium
    # This example is best run if BrowserAgent_Enable is true in config.
    if settings and settings.browser_agent_enable:
        asyncio.run(main_browser_agent_example())
    else:
        print("BrowserAgent is not enabled in settings, or settings failed to load. Skipping example.")
        print("To run, set BrowserAgent_Enable = True in config.ini and ensure Vertex AI is configured.")