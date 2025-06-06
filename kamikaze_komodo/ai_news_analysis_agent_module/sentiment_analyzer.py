# FILE: kamikaze_komodo/ai_news_analysis_agent_module/sentiment_analyzer.py
from typing import List, Dict, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.core.models import NewsArticle
from kamikaze_komodo.config.settings import settings # Import global settings

logger = get_logger(__name__)

# Pydantic model for structured sentiment output
class SentimentAnalysisOutput(BaseModel):
    sentiment_label: str = Field(description="The overall sentiment (e.g., 'very bullish', 'bullish', 'neutral', 'bearish', 'very bearish', 'mixed').")
    sentiment_score: float = Field(description="A numerical score from -1.0 (very negative) to 1.0 (very positive). Neutral is 0.0.")
    key_themes: Optional[List[str]] = Field(default_factory=list, description="List of key themes or topics identified in the text related to sentiment.")
    confidence: Optional[float] = Field(description="Confidence score of the sentiment analysis (0.0 to 1.0).")

class SentimentAnalyzer:
    """
    Analyzes text for sentiment using a configured LLM via Langchain.
    Supports Google Vertex AI.
    """
    def __init__(self):
        if not settings:
            logger.critical("Settings not loaded. SentimentAnalyzer cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.llm_provider = settings.sentiment_llm_provider
        self.llm: Any = None # Will be initialized in _initialize_llm

        self._initialize_llm()

        # Define a structured prompt for sentiment analysis
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are an expert financial sentiment analyst specializing in cryptocurrency markets. "
                 "Analyze the provided text for its sentiment towards the cryptocurrency or market mentioned. "
                 "Consider factors like news events, market reactions, technological developments, and regulatory news. "
                 "Your output MUST be in JSON format, adhering to the following Pydantic model structure (SentimentAnalysisOutput): "
                 "```json\n"
                 "{\n"
                 "  \"sentiment_label\": \"<label: 'very bullish'|'bullish'|'neutral'|'bearish'|'very bearish'|'mixed'>\",\n"
                 "  \"sentiment_score\": <score_float: -1.0 to 1.0>,\n"
                 "  \"key_themes\": [\"<theme1>\", \"<theme2>\"],\n" # Optional
                 "  \"confidence\": <confidence_float: 0.0 to 1.0>\n" # Optional
                 "}\n"
                 "```"
                 "sentiment_score should range from -1.0 (very bearish/negative) to 1.0 (very bullish/positive). Neutral is 0.0. "
                 "key_themes should highlight important topics influencing the sentiment. confidence is your perceived accuracy of this analysis (0.0 to 1.0)."
                 ),
                ("human", "Please analyze the sentiment of the following text regarding {asset_context}:\n\n---\n{text_to_analyze}\n---"),
            ]
        )
        self.output_parser = JsonOutputParser(pydantic_object=SentimentAnalysisOutput)
        self.chain = self.prompt_template | self.llm | self.output_parser

    def _initialize_llm(self):
        logger.info(f"Initializing LLM for SentimentAnalyzer with provider: {self.llm_provider}")
        if self.llm_provider == "VertexAI":
            if not settings.vertex_ai_project_id or not settings.vertex_ai_location:
                logger.error("Vertex AI project ID or location is not configured in settings.py. Sentiment analysis will not work.")
                raise ValueError("Vertex AI project ID or location missing.")
            try:
                from langchain_google_vertexai import ChatVertexAI # Correct import
                self.llm = ChatVertexAI(
                    project=settings.vertex_ai_project_id,
                    location=settings.vertex_ai_location,
                    model_name=settings.vertex_ai_sentiment_model_name,
                    temperature=0.1, # Low temperature for more factual/consistent sentiment
                    # max_output_tokens=1024, # Optional: if needed for longer summaries/themes
                )
                logger.info(f"SentimentAnalyzer initialized with Vertex AI model: {settings.vertex_ai_sentiment_model_name}")
                logger.info("Ensure Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS) are set in your environment.")
            except ImportError:
                logger.error("langchain-google-vertexai is not installed. Please install it: pip install langchain-google-vertexai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI LLM ({settings.vertex_ai_sentiment_model_name}): {e}", exc_info=True)
                raise
        else:
            logger.error(f"Unsupported LLM provider: {self.llm_provider}")
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        if self.llm is None:
            logger.error("LLM could not be initialized.")
            raise ValueError("LLM initialization failed.")

    async def analyze_sentiment_structured(self, text: str, asset_context: str = "the market") -> Optional[SentimentAnalysisOutput]:
        """
        Analyzes text and returns a structured sentiment analysis including score and label.
        """
        if not text or not text.strip():
            logger.warning("No text provided for sentiment analysis.")
            return None

        if self.llm is None:
            logger.error("LLM not initialized. Cannot analyze sentiment.")
            return None

        logger.debug(f"Analyzing sentiment for text (context: {asset_context}): '{text[:200]}...'")
        try:
            # Max input tokens for gemini-2.5-flash-preview is high, but let's be reasonable.
            # Prompt itself consumes tokens. Max 30k chars ~ 7.5k tokens for text.
            max_chars = 30000
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} for sentiment analysis.")
                text = text[:max_chars]

            response_dict = await self.chain.ainvoke({"text_to_analyze": text, "asset_context": asset_context})

            # The output_parser should already return a SentimentAnalysisOutput object if successful
            # If it's a dict, it means JsonOutputParser might not have directly instantiated the Pydantic model
            # or the LLM didn't return perfect JSON matching the Pydantic model structure.
            if isinstance(response_dict, dict):
                try:
                    # Attempt to create Pydantic model from dict for validation and type safety
                    validated_output = SentimentAnalysisOutput(**response_dict)
                    logger.info(f"Structured sentiment for '{asset_context}': Label: {validated_output.sentiment_label}, Score: {validated_output.sentiment_score:.2f}, Confidence: {validated_output.confidence}")
                    return validated_output
                except Exception as p_exc:
                    logger.error(f"Pydantic validation failed for LLM JSON output: {response_dict}. Error: {p_exc}", exc_info=True)
                    return None
            elif isinstance(response_dict, SentimentAnalysisOutput): # Already a Pydantic object
                   logger.info(f"Structured sentiment for '{asset_context}': Label: {response_dict.sentiment_label}, Score: {response_dict.sentiment_score:.2f}, Confidence: {response_dict.confidence}")
                   return response_dict
            else:
                logger.error(f"Unexpected structured sentiment analysis output type: {type(response_dict)}. Content: {str(response_dict)[:500]}")
                return None
        except Exception as e:
            logger.error(f"Error during structured sentiment analysis with {self.llm_provider} model: {e}", exc_info=True)
            return None

    async def get_sentiment_for_article(self, article: NewsArticle, asset_context: Optional[str] = None) -> NewsArticle:
        """
        Analyzes sentiment for a NewsArticle object and updates its sentiment fields.
        Uses article title and summary/content.
        """
        if not asset_context and article.related_symbols:
            asset_context = ", ".join(article.related_symbols)
        elif not asset_context:
            # Try to infer from title if no symbols
            if "bitcoin" in article.title.lower() or "btc" in article.title.lower():
                asset_context = "Bitcoin"
            elif "ethereum" in article.title.lower() or "eth" in article.title.lower():
                asset_context = "Ethereum"
            else:
                asset_context = "the cryptocurrency market"

        text_to_analyze = article.title
        if article.summary:
            text_to_analyze += "\n\n" + article.summary
        elif article.content: # Fallback to content if no summary
            text_to_analyze += "\n\n" + article.content

        if not text_to_analyze.strip():
            logger.warning(f"No text content found in article {article.id} to analyze.")
            return article # Return original article if no text

        sentiment_result = await self.analyze_sentiment_structured(text_to_analyze, asset_context=asset_context)

        if sentiment_result:
            article.sentiment_label = sentiment_result.sentiment_label
            article.sentiment_score = sentiment_result.sentiment_score
            article.key_themes = sentiment_result.key_themes
            article.sentiment_confidence = sentiment_result.confidence
            # article.raw_llm_response can store the full dict if needed for debugging
            # article.raw_llm_response = sentiment_result.model_dump()
        return article

# Example Usage
async def main_sentiment_example():
    """ Example of using the SentimentAnalyzer """
    if not settings or not settings.vertex_ai_project_id:
        logger.error("Vertex AI settings (Project ID) not loaded for sentiment example. Set GOOGLE_APPLICATION_CREDENTIALS env var.")
        return

    try:
        analyzer = SentimentAnalyzer()
    except Exception as e:
        logger.error(f"Could not start SentimentAnalyzer: {e}")
        return

    example_texts = [
        ("Bitcoin surges past $70,000, analysts predict further upside due to ETF inflows and positive market structure.", "Bitcoin"),
        ("Regulatory crackdown imminent? SEC chair issues stark warning on crypto staking, leading to market jitters.", "Cryptocurrency Regulation"),
        ("Ethereum's Dencun upgrade successfully goes live on mainnet, promising significantly lower fees for Layer 2 solutions and boosting scalability.", "Ethereum"),
        ("The crypto market remains flat this week with low volatility and trading volume, investors seem hesitant.", "the crypto market"),
        ("Solana's network outage causes temporary panic, but recovery was swift. Developers are addressing the root cause.", "Solana")
    ]

    for text, context in example_texts:
        logger.info(f"\n--- Analyzing text for '{context}' ---")
        logger.info(f"Text: {text}")
        result = await analyzer.analyze_sentiment_structured(text, asset_context=context)
        if result:
            logger.info(f"  Sentiment Label: {result.sentiment_label}")
            logger.info(f"  Sentiment Score: {result.sentiment_score:.3f}")
            logger.info(f"  Key Themes: {result.key_themes}")
            logger.info(f"  Confidence: {result.confidence}")
        else:
            logger.warning("  Failed to get structured sentiment analysis.")

    sample_article = NewsArticle(
        id="test_article_sol_123",
        url="http://example.com/news_sol_1",
        title="Solana Ecosystem Sees Major Investment for DeFi Growth",
        summary="The Solana Foundation has announced a new $100 million fund dedicated to fostering DeFi projects on its blockchain. This move is expected to attract more developers and users, with SOL token price reacting positively.",
        source="Crypto News Daily",
        related_symbols=["SOL", "Solana"]
    )
    logger.info(f"\n--- Analyzing NewsArticle: {sample_article.title} ---")
    updated_article = await analyzer.get_sentiment_for_article(sample_article)
    logger.info(f"  Analyzed Article Sentiment: Label='{updated_article.sentiment_label}', Score={updated_article.sentiment_score}, Themes: {updated_article.key_themes}")

if __name__ == "__main__":
    import asyncio
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
    # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    asyncio.run(main_sentiment_example())