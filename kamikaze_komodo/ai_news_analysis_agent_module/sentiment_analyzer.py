# kamikaze_komodo/ai_news_analysis_agent_module/sentiment_analyzer.py
from typing import List, Dict, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field as PydanticField

# Assuming local Ollama setup as per testOllama.py and testBrowserUse.py
from langchain_ollama import ChatOllama

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.core.models import NewsArticle

logger = get_logger(__name__)

# Pydantic model for structured sentiment output
class SentimentAnalysisOutput(BaseModel):
    sentiment_label: str = PydanticField(description="The overall sentiment (e.g., 'positive', 'negative', 'neutral', 'mixed', 'bullish', 'bearish').")
    sentiment_score: float = PydanticField(description="A numerical score from -1.0 (very negative) to 1.0 (very positive). Neutral is 0.0.")
    key_themes: Optional[List[str]] = PydanticField(description="List of key themes or topics identified in the text related to sentiment.")
    confidence: Optional[float] = PydanticField(description="Confidence score of the sentiment analysis (0.0 to 1.0).")


class SentimentAnalyzer:
    """
    Analyzes text for sentiment using a local LLM via Langchain and Ollama.
    """
    def __init__(self, llm_model_name: str = "gemma3:12b", ollama_base_url: Optional[str] = None):
        self.llm_model_name = llm_model_name
        try:
            llm_params = {"model": self.llm_model_name, "temperature": 0.1}
            if ollama_base_url:
                llm_params["base_url"] = ollama_base_url
            
            self.llm = ChatOllama(**llm_params)
            logger.info(f"SentimentAnalyzer initialized with LLM: {self.llm_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM ({self.llm_model_name}): {e}. Ensure Ollama is running and model is pulled.")
            raise

        # Define a more structured prompt for sentiment analysis
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are an expert financial sentiment analyst specializing in cryptocurrency markets. "
                 "Analyze the provided text for its sentiment towards the cryptocurrency or market mentioned. "
                 "Consider factors like news events, market reactions, technological developments, and regulatory news. "
                 "Your output MUST be in JSON format, following this structure: "
                 "{{\"sentiment_label\": \"<label>\", \"sentiment_score\": <score_float>, \"key_themes\": [\"<theme1>\", \"<theme2>\"], \"confidence\": <confidence_float>}}."
                 "Possible sentiment_labels: 'very bullish', 'bullish', 'neutral', 'bearish', 'very bearish', 'mixed'. "
                 "sentiment_score should range from -1.0 (very bearish/negative) to 1.0 (very bullish/positive). Neutral is 0.0. "
                 "key_themes should highlight important topics influencing the sentiment. confidence is your perceived accuracy of this analysis (0.0 to 1.0)."
                 ),
                ("human", "Please analyze the sentiment of the following text regarding {asset_context}:\n\n---\n{text_to_analyze}\n---"),
            ]
        )
        
        # Output parser for structured JSON
        self.output_parser = JsonOutputParser(pydantic_object=SentimentAnalysisOutput)
        
        # For simpler string output if JSON fails or for basic sentiment
        self.string_output_parser = StrOutputParser()

        self.chain = self.prompt_template | self.llm | self.output_parser


    async def analyze_sentiment_structured(self, text: str, asset_context: str = "the market") -> Optional[SentimentAnalysisOutput]:
        """
        Analyzes text and returns a structured sentiment analysis including score and label.
        """
        if not text or not text.strip():
            logger.warning("No text provided for sentiment analysis.")
            return None
        
        logger.debug(f"Analyzing sentiment for text (context: {asset_context}): '{text[:200]}...'")
        try:
            # Shorten text if too long for the model's context window (gemma3 typically 8k tokens)
            # A more robust solution would use token counting. This is a simple character limit.
            max_chars = 12000 # Approx 3k-4k tokens, depends on text.
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} for sentiment analysis.")
                text = text[:max_chars]

            response = await self.chain.ainvoke({"text_to_analyze": text, "asset_context": asset_context})
            
            # The response should already be a dict parsed into SentimentAnalysisOutput by JsonOutputParser
            if isinstance(response, dict): # If JsonOutputParser worked
                 # Validate with Pydantic model if not automatically done (JsonOutputParser should do it)
                try:
                    validated_output = SentimentAnalysisOutput(**response)
                    logger.info(f"Structured sentiment analysis for '{asset_context}': Label: {validated_output.sentiment_label}, Score: {validated_output.sentiment_score:.2f}, Confidence: {validated_output.confidence}")
                    return validated_output
                except Exception as p_exc: # PydanticValidationError
                    logger.error(f"Pydantic validation failed for LLM JSON output: {response}. Error: {p_exc}")
                    return None
            else: # Fallback or unexpected output type
                logger.error(f"Unexpected structured sentiment analysis output type: {type(response)}. Content: {response}")
                return None

        except Exception as e:
            logger.error(f"Error during structured sentiment analysis with LLM {self.llm_model_name}: {e}", exc_info=True)
            # Fallback: try a simpler prompt if structured fails? Or just return None.
            return None

    async def get_sentiment_for_article(self, article: NewsArticle, asset_context: Optional[str] = None) -> NewsArticle:
        """
        Analyzes sentiment for a NewsArticle object and updates its sentiment fields.
        Uses article title and summary/content.
        """
        if not asset_context and article.related_symbols:
            asset_context = ", ".join(article.related_symbols)
        elif not asset_context:
            asset_context = "the cryptocurrency market"

        text_to_analyze = article.title
        if article.summary:
            text_to_analyze += "\n\n" + article.summary
        elif article.content:
            text_to_analyze += "\n\n" + article.content

        if not text_to_analyze.strip():
            logger.warning(f"No text content found in article {article.id} to analyze.")
            return article

        sentiment_result = await self.analyze_sentiment_structured(text_to_analyze, asset_context=asset_context)

        if sentiment_result:
            article.sentiment_label = sentiment_result.sentiment_label
            article.sentiment_score = sentiment_result.sentiment_score
            # article.custom_fields = article.custom_fields or {} # If using custom fields
            # article.custom_fields['sentiment_key_themes'] = sentiment_result.key_themes
            # article.custom_fields['sentiment_confidence'] = sentiment_result.confidence
        return article

# Example Usage
async def main_sentiment_example(settings_obj):
    """ Example of using the SentimentAnalyzer """
    if not settings_obj:
        logger.error("Settings not loaded for sentiment example.")
        return

    try:
        analyzer = SentimentAnalyzer(llm_model_name=settings_obj.sentiment_llm_model)
    except Exception as e:
        logger.error(f"Could not start SentimentAnalyzer: {e}")
        return

    example_texts = [
        ("Bitcoin surges past $70,000, analysts predict further upside due to ETF inflows.", "Bitcoin"),
        ("Regulatory crackdown imminent? SEC chair issues stark warning on crypto staking.", "Cryptocurrency Regulation"),
        ("Ethereum's Dencun upgrade goes live, promising lower fees for Layer 2s.", "Ethereum"),
        ("The crypto market remains flat this week with low volatility and trading volume.", "the crypto market")
    ]

    for text, context in example_texts:
        logger.info(f"\n--- Analyzing text for '{context}' ---")
        logger.info(f"Text: {text}")
        result = await analyzer.analyze_sentiment_structured(text, asset_context=context)
        if result:
            logger.info(f"Sentiment Label: {result.sentiment_label}")
            logger.info(f"Sentiment Score: {result.sentiment_score:.3f}")
            logger.info(f"Key Themes: {result.key_themes}")
            logger.info(f"Confidence: {result.confidence}")
        else:
            logger.warning("Failed to get structured sentiment analysis.")

    # Example with NewsArticle
    sample_article = NewsArticle(
        id="test_article_123",
        url="http://example.com/news1",
        title="Groundbreaking Partnership Announced for Solana Ecosystem Development",
        summary="Solana Foundation today announced a strategic partnership with a major tech firm to boost its ecosystem infrastructure and developer tools. The SOL token reacted positively, climbing 5% in the last hour.",
        source="Test News Provider",
        related_symbols=["SOL", "Solana"]
    )
    logger.info(f"\n--- Analyzing NewsArticle: {sample_article.title} ---")
    updated_article = await analyzer.get_sentiment_for_article(sample_article)
    logger.info(f"Analyzed Article Sentiment: Label='{updated_article.sentiment_label}', Score={updated_article.sentiment_score}")


# if __name__ == "__main__":
#     # from kamikaze_komodo.config.settings import settings
#     # if settings:
#     #    asyncio.run(main_sentiment_example(settings))
#     # else:
#     #    print("Run this example from the project root or ensure settings are available.")
#     pass