
import aiohttp
import json
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

from cipher_edge.core.models import NewsArticle
from cipher_edge.app_logger import get_logger

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)


class NewsAnalysisResult(BaseModel):
    """A structured model for holding the results of a news article analysis."""
    summary: str = Field(description="A concise, neutral summary of the article in 2-3 sentences.")
    sentiment_score: float = Field(description="A float from -1.0 (very bearish) to 1.0 (very bullish).")
    sentiment_label: str = Field(description="A string label: 'bullish', 'bearish', or 'neutral'.")
    key_themes: List[str] = Field(description="A list of 2-4 key themes or topics discussed.")
    related_symbols: List[str] = Field(description="A list of relevant cryptocurrency ticker symbols (e.g., ['BTC', 'ETH']).")


class NewsProcessor:
    """
    Processes a news article by fetching its full content, extracting text,
    and using an LLM to analyze and enrich it with a guaranteed structure.
    """
    def __init__(self, model_name: str = "qwen2:7b"):
        """
        Initializes the NewsProcessor, setting up the LLM and HTTP session.
        """
        llm_base = ChatOllama(model=model_name, temperature=0.1)
        self.structured_llm = llm_base.with_structured_output(NewsAnalysisResult)

        self.session: Optional[aiohttp.ClientSession] = None
        self.prompt_template = self._create_prompt_template()
        logger.info(f"NewsProcessor initialized with structured output for model: {model_name}")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Creates the prompt template. With structured output, we only need to
        define the task, not the specific JSON format.
        """
        system_message = (
            "You are an expert financial analyst specializing in cryptocurrency. "
            "Your task is to analyze a news article and extract the requested information."
        )
        
        human_template = "Please analyze the following news article text:\n\n{article_text}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template),
        ])

    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns the aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={'User-Agent': 'CipherEdge/1.0'})
        return self.session

    def _extract_article_text(self, html_content: str) -> str:
        """
        Extracts the main text content from the article's HTML using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        main_content = soup.find('article') or soup.find('main') or soup.body
        if not main_content:
            return ""
        paragraphs = main_content.find_all('p')
        full_text = ' '.join(p.get_text() for p in paragraphs)
        return full_text.strip()

    async def process_article(self, article: NewsArticle) -> NewsArticle:
        """
        The main pipeline for processing a single news article.
        """
        logger.info(f"Processing article: {article.url}")
        session = await self._get_session()
        
        try:
            # 1. Fetch HTML content
            async with session.get(article.url, timeout=20) as response:
                response.raise_for_status()
                html = await response.text()

            article_text = self._extract_article_text(html)
            if not article_text or len(article_text) < 200:
                logger.warning(f"Extracted text for {article.url} is too short. Skipping LLM analysis.")
                return article

            chain = self.prompt_template | self.structured_llm
            
            analysis_data: NewsAnalysisResult = await chain.ainvoke({"article_text": article_text[:8000]})
            
            if analysis_data:
                logger.info(f"Successfully analyzed article: {article.title}")
                article.summary = analysis_data.summary
                article.sentiment_score = analysis_data.sentiment_score
                article.sentiment_label = analysis_data.sentiment_label
                article.key_themes = analysis_data.key_themes
                article.related_symbols = analysis_data.related_symbols
                article.raw_llm_response = analysis_data.dict()

        except Exception as e:
            logger.error(f"Failed to process article {article.url}: {e}", exc_info=False)
            
        return article

    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("NewsProcessor HTTP session closed.")