import asyncio
import aiohttp
import feedparser
import hashlib
from datetime import datetime, timezone
from typing import Callable, Awaitable, Set, Dict, Optional, List

from cipher_edge.config.settings import settings
from cipher_edge.app_logger import get_logger
from cipher_edge.core.models import NewsArticle
from cipher_edge.ai_market_intelligence.processor import NewsProcessor

logger = get_logger(__name__)

news_processor = NewsProcessor()


async def process_news_event(article: NewsArticle):
    """
    Callback that receives a new article, processes it to get full-text
    analysis, and then logs the enriched result.
    """
    logger.info(f"Listener found article: '{article.title}'. Handing to processor...")

    # Use the global processor instance to analyze the article
    enriched_article = await news_processor.process_article(article)

    # Print the enriched data to see the result
    if enriched_article.summary:
        print("\n--- Article Analysis Complete ---")
        print(f"Title: {enriched_article.title}")
        print(f"Source: {enriched_article.source}")
        print(f"URL: {enriched_article.url}")
        print(f"Summary: {enriched_article.summary}")
        print(f"Sentiment: {enriched_article.sentiment_label} ({enriched_article.sentiment_score})")
        print(f"Themes: {enriched_article.key_themes}")
        print(f"Symbols: {enriched_article.related_symbols}")
        print("-------------------------------------\n")
    else:
        logger.warning(f"Article '{article.title}' could not be fully analyzed.")


class NewsListener:
    """
    Asynchronously monitors multiple RSS feeds for new articles and triggers a
    callback function when a new item is detected.
    """

    def __init__(self, callback: Callable[[NewsArticle], Awaitable[None]]):
        """
        Initializes the NewsListener.

        Args:
            callback: An async function to be called with a `NewsArticle`
                      object when a new article is found.
        """
        if not settings:
            logger.critical("Settings not loaded. NewsListener cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.callback = callback
        self.rss_feeds: List[Dict[str, str]] = settings.rss_feeds
        self.check_interval: int = settings.news_listener_check_interval
        self.seen_articles: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_first_run = True

    @staticmethod
    def _get_article_id(url: str) -> str:
        """Creates a consistent SHA256 hash for a URL to use as a unique ID."""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    async def _fetch_and_parse_feed(self, feed_info: Dict[str, str], is_initial_run: bool = False):
        """
        Fetches a single RSS feed, parses it, and processes new entries.
        """
        name, url = feed_info['name'], feed_info['url']
        try:
            if not self.session or self.session.closed:
                logger.error("AIOHTTP session is not active. Cannot fetch feed.")
                return

            headers = {'User-Agent': 'CipherEdge/1.0'}
            async with self.session.get(url, timeout=15, headers=headers) as response:
                response.raise_for_status()
                content = await response.text()

            parsed_feed = feedparser.parse(content)

            if parsed_feed.bozo:
                logger.warning(f"Feed '{name}' ({url}) may be malformed. Details: {parsed_feed.bozo_exception}")

            for entry in parsed_feed.entries:
                if not hasattr(entry, 'link') or not hasattr(entry, 'title'):
                    continue

                article_id = self._get_article_id(entry.link)

                if article_id not in self.seen_articles:
                    self.seen_articles.add(article_id)

                    if not is_initial_run:
                        logger.info(f"New article from '{name}': {entry.title}")

                        publication_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                publication_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                            except (TypeError, ValueError):
                                logger.warning(f"Could not parse publication date for article: {entry.link}")

                        article = NewsArticle(
                            id=article_id,
                            url=entry.link,
                            title=entry.title,
                            publication_date=publication_date,
                            source=name,
                            summary=entry.summary if hasattr(entry, 'summary') else None
                        )
                        await self.callback(article)

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching feed '{name}' ({url}): {e}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching feed '{name}' ({url}).")
        except Exception as e:
            logger.error(f"An unexpected error occurred processing feed '{name}' ({url}): {e}", exc_info=True)

    async def start_listening(self):
        """
        Starts the continuous process of checking RSS feeds for new articles.
        """
        self.session = aiohttp.ClientSession()

        if self._is_first_run:
            logger.info("NewsListener starting: Populating initial set of articles to avoid spam...")
            initial_tasks = [self._fetch_and_parse_feed(feed, is_initial_run=True) for feed in self.rss_feeds]
            await asyncio.gather(*initial_tasks)
            self._is_first_run = False
            logger.info(f"Initial population complete. Found {len(self.seen_articles)} existing articles.")

        logger.info(f"Starting to monitor {len(self.rss_feeds)} RSS feeds. Checking every {self.check_interval} seconds.")

        while True:
            try:
                tasks = [self._fetch_and_parse_feed(feed) for feed in self.rss_feeds]
                await asyncio.gather(*tasks)
                logger.debug(f"Completed a check of all RSS feeds. Waiting for {self.check_interval} seconds.")
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                logger.info("News listener task was cancelled.")
                break

    async def stop(self):
        """
        Gracefully stops the listener by closing the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("News listener's HTTP session has been closed.")


if __name__ == "__main__":
    async def main():
        if not settings or not settings.notification_listener_enable:
            logger.info("News listener is disabled in config.ini. Exiting.")
            return

        listener = NewsListener(callback=process_news_event)
        try:
            await listener.start_listening()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            await listener.stop()
            await news_processor.close() # Clean up the processor's session
            
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"An error occurred in the main listener loop: {e}", exc_info=True)