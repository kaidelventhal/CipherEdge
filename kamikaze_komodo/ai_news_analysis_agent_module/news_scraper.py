# kamikaze_komodo/ai_news_analysis_agent_module/news_scraper.py
import asyncio
import feedparser
import newspaper # type: ignore
import httpx
from typing import List, Optional, Dict, Any
# from bs4 import BeautifulSoup # Keep for potential future direct HTML parsing needs
from kamikaze_komodo.core.models import NewsArticle
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone, timedelta
from kamikaze_komodo.config.settings import settings # Import global settings

logger = get_logger(__name__)

class NewsScraper:
    """
    Scrapes news from specified sources (RSS feeds, websites).
    """
    def __init__(self):
        if not settings:
            logger.critical("Settings not loaded. NewsScraper cannot be initialized.")
            raise ValueError("Settings not loaded.")

        scraper_config = settings.get_news_scraper_config()
        self.rss_feeds: List[Dict[str, str]] = scraper_config.get("rss_feeds", [])
        self.websites_to_scrape: List[Dict[str, str]] = scraper_config.get("websites", []) # For Newspaper3k or custom BS4

        if not self.rss_feeds and not self.websites_to_scrape:
            logger.warning("NewsScraper initialized, but no RSS feeds or websites are configured in settings.")
        else:
            logger.info(f"NewsScraper initialized. RSS feeds: {len(self.rss_feeds)}, Websites: {len(self.websites_to_scrape)}")
            if self.rss_feeds:
                logger.debug(f"Configured RSS Feeds: {[feed['name'] for feed in self.rss_feeds]}")

    async def _fetch_url_content(self, url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 KamikazeKomodoBot/1.0'
                }
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.text
        except httpx.RequestError as e:
            logger.error(f"HTTP request error fetching URL {url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error fetching URL {url}: {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e_gen:
            logger.error(f"Generic error fetching URL {url}: {e_gen}", exc_info=True)
        return None

    async def scrape_rss_feed(self, feed_name: str, feed_url: str, limit: int = 15) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        logger.info(f"Scraping RSS feed: {feed_name} from {feed_url}")

        feed_content = await self._fetch_url_content(feed_url)
        if not feed_content:
            logger.warning(f"Could not fetch content for RSS feed {feed_name} ({feed_url}). Skipping.")
            return articles

        try:
            loop = asyncio.get_event_loop()
            # feedparser is synchronous
            parsed_feed = await loop.run_in_executor(None, feedparser.parse, feed_content)

            if parsed_feed.bozo:
                logger.warning(f"Error parsing RSS feed {feed_name} ({feed_url}): {parsed_feed.bozo_exception}")

            if not parsed_feed.entries:
                logger.info(f"No entries found in RSS feed: {feed_name} ({feed_url}).")
                return articles

            for entry in parsed_feed.entries[:limit]:
                title = entry.get("title")
                link = entry.get("link")
                if not title or not link:
                    logger.debug(f"Skipping entry with missing title or link in {feed_name}: {entry.get('id', 'N/A')}")
                    continue

                published_time_struct = entry.get("published_parsed")
                updated_time_struct = entry.get("updated_parsed")

                pub_date: Optional[datetime] = None
                time_struct_to_use = published_time_struct or updated_time_struct

                if time_struct_to_use:
                    try:
                        pub_date = datetime(*time_struct_to_use[:6], tzinfo=timezone.utc)
                    except Exception as e_date:
                        logger.warning(f"Could not parse date for article '{title}' from {feed_name}: {time_struct_to_use}, error: {e_date}")

                # Fallback if date parsing fails or not present
                if pub_date is None:
                    pub_date = datetime.now(timezone.utc) # Use retrieval time as a last resort
                    logger.debug(f"Using current time as publication date for '{title}' from {feed_name} due to missing/unparseable date.")


                article_id = link # Use URL as a unique ID

                content_summary = entry.get("summary") or entry.get("description")

                # Attempt to extract related symbols from title or summary (basic)
                related_symbols = []
                text_for_symbols = (title + " " + (content_summary if content_summary else "")).lower()
                # This is very basic; a proper NER would be better
                common_crypto_symbols = {"btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "ada", "cardano", "doge", "shib"}
                for sym in common_crypto_symbols:
                    if sym in text_for_symbols:
                        related_symbols.append(sym.upper())

                articles.append(NewsArticle(
                    id=article_id,
                    url=link,
                    title=title,
                    publication_date=pub_date,
                    retrieval_date=datetime.now(timezone.utc),
                    source=feed_name,
                    content=None, # Full content fetch can be added here or later by newspaper3k
                    summary=content_summary,
                    related_symbols=list(set(related_symbols)) # Unique symbols
                ))
            logger.info(f"Found {len(articles)} articles from RSS feed: {feed_name}")
        except Exception as e:
            logger.error(f"Failed to process RSS feed {feed_name} ({feed_url}): {e}", exc_info=True)
        return articles

    async def scrape_website_with_newspaper(self, site_name: str, site_url: str, limit_articles: int = 5) -> List[NewsArticle]:
        """Scrapes a website using Newspaper3k. Be mindful of terms of service."""
        articles_data: List[NewsArticle] = []
        logger.info(f"Scraping website: {site_name} ({site_url}) with Newspaper3k (limit: {limit_articles})")

        # Newspaper3k config
        config_np = newspaper.Config()
        config_np.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 KamikazeKomodoBot/1.0'
        config_np.request_timeout = 15
        config_np.memoize_articles = False # Disable caching for fresh data
        config_np.fetch_images = False # Don't need images
        config_np.verbose = False # newspaper's own verbosity

        try:
            loop = asyncio.get_event_loop()
            paper = await loop.run_in_executor(None, newspaper.build, site_url, config_np)

            count = 0
            for article_raw in paper.articles:
                if count >= limit_articles:
                    break
                try:
                    # Download and parse article content
                    await loop.run_in_executor(None, article_raw.download)
                    if not article_raw.is_downloaded:
                        logger.warning(f"Failed to download article: {article_raw.url} from {site_name}")
                        continue
                    await loop.run_in_executor(None, article_raw.parse)

                    title = article_raw.title
                    url = article_raw.url
                    if not title or not url:
                        logger.debug(f"Skipping article with no title/url from {site_name}")
                        continue

                    content = article_raw.text
                    summary_np = article_raw.summary # newspaper3k summary

                    pub_date_dt = article_raw.publish_date
                    if pub_date_dt and pub_date_dt.tzinfo is None:
                        pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc) # Assume UTC if naive, or local if known
                    elif pub_date_dt is None:
                        pub_date_dt = datetime.now(timezone.utc) # Fallback

                    related_symbols_np = []
                    text_for_symbols_np = (title + " " + (summary_np if summary_np else "") + " " + (content if content else "")).lower()
                    common_crypto_symbols_np = {"btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "ada", "cardano", "doge", "shib"}
                    for sym_np in common_crypto_symbols_np:
                        if sym_np in text_for_symbols_np:
                            related_symbols_np.append(sym_np.upper())

                    articles_data.append(NewsArticle(
                        id=url, url=url, title=title,
                        publication_date=pub_date_dt,
                        retrieval_date=datetime.now(timezone.utc),
                        source=site_name,
                        content=content if content else None,
                        summary=summary_np if summary_np else None,
                        related_symbols=list(set(related_symbols_np))
                    ))
                    count += 1
                    logger.debug(f"Successfully scraped: {url} from {site_name}")
                except Exception as e_article:
                    logger.warning(f"Error processing article {article_raw.url} from {site_name} with Newspaper3k: {e_article}", exc_info=True)

            logger.info(f"Scraped {len(articles_data)} articles from {site_name} using Newspaper3k.")
        except Exception as e:
            logger.error(f"Failed to scrape website {site_name} ({site_url}) with Newspaper3k: {e}", exc_info=True)
        return articles_data

    async def scrape_all(self, limit_per_source: int = 10, since_hours_rss: Optional[int] = 24) -> List[NewsArticle]:
        """
        Scrapes all configured RSS feeds and websites.
        For RSS, optionally filters articles published within `since_hours_rss`.
        """
        all_articles: List[NewsArticle] = []

        # Scrape RSS Feeds
        rss_tasks = []
        if self.rss_feeds:
            for feed_info in self.rss_feeds:
                rss_tasks.append(self.scrape_rss_feed(feed_info['name'], feed_info['url'], limit=limit_per_source))

            rss_results_list = await asyncio.gather(*rss_tasks, return_exceptions=True)
            for result in rss_results_list:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"RSS scraping task failed: {result}", exc_info=True) # Log exception details
        else:
            logger.info("No RSS feeds configured to scrape.")

        # Filter RSS articles by publication date if since_hours_rss is provided
        if since_hours_rss is not None:
            cutoff_date = datetime.now(timezone.utc) - timedelta(hours=since_hours_rss)
            filtered_articles = []
            for article in all_articles:
                if article.publication_date and article.publication_date >= cutoff_date:
                    filtered_articles.append(article)
                elif not article.publication_date: # If no pub date, include it (conservative)
                    filtered_articles.append(article)
            count_removed = len(all_articles) - len(filtered_articles)
            if count_removed > 0:
                logger.info(f"Filtered out {count_removed} RSS articles older than {since_hours_rss} hours.")
            all_articles = filtered_articles

        # Scrape Websites (e.g., using Newspaper3k) - typically gets latest, less date control
        website_tasks = []
        if self.websites_to_scrape:
            for site_info in self.websites_to_scrape:
                website_tasks.append(self.scrape_website_with_newspaper(site_info['name'], site_info['url'], limit_articles=limit_per_source))

            website_results_list = await asyncio.gather(*website_tasks, return_exceptions=True)
            for result in website_results_list:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Website scraping task failed: {result}", exc_info=True)
        else:
            logger.info("No direct websites configured to scrape with Newspaper3k.")

        # Deduplicate articles by URL (ID)
        unique_articles_dict: Dict[str, NewsArticle] = {}
        for article in all_articles:
            if article.id not in unique_articles_dict:
                unique_articles_dict[article.id] = article
            else: # If duplicate, prefer the one with more content or later retrieval
                existing_article = unique_articles_dict[article.id]
                if (article.content and not existing_article.content) or \
                   (article.retrieval_date > existing_article.retrieval_date):
                    unique_articles_dict[article.id] = article

        unique_articles_list = sorted(list(unique_articles_dict.values()), key=lambda x: x.publication_date or x.retrieval_date, reverse=True)

        logger.info(f"Total unique articles scraped from all sources: {len(unique_articles_list)}")
        return unique_articles_list

async def main_scraper_example():
    if not settings or not settings.news_scraper_enable:
        logger.info("NewsScraper is not enabled in settings or settings not loaded.")
        return

    scraper = NewsScraper()

    # Scrape all configured sources, limiting to 5 articles per source,
    # and only RSS articles from the last 48 hours
    all_scraped_articles = await scraper.scrape_all(limit_per_source=5, since_hours_rss=48)

    logger.info(f"--- All Scraped Articles ({len(all_scraped_articles)}) ---")
    if not all_scraped_articles:
        logger.info("No articles were scraped.")
        return

    for i, article in enumerate(all_scraped_articles[:10]): # Log details for first 10
        logger.info(f"{i+1}. Source: {article.source}, Title: {article.title}")
        logger.info(f"    URL: {article.url}")
        logger.info(f"    Date: {article.publication_date}, Retrieved: {article.retrieval_date}")
        logger.info(f"    Symbols: {article.related_symbols}")
        if article.summary:
            logger.info(f"    Summary: {article.summary[:150]}...")
        # if article.content: # Content can be very long
            # logger.info(f" Content Preview: {article.content[:100]}...")

    # Example: Store articles in DB
    if all_scraped_articles:
        from kamikaze_komodo.data_handling.database_manager import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.store_news_articles(all_scraped_articles)
        logger.info(f"Stored {len(all_scraped_articles)} articles in the database.")

        # Retrieve and show some from DB
        retrieved = db_manager.retrieve_news_articles(limit=5)
        logger.info(f"--- Retrieved {len(retrieved)} articles from DB ---")
        for art_db in retrieved:
            logger.info(f"DB: {art_db.title} (Source: {art_db.source}, Date: {art_db.publication_date})")
        db_manager.close()

if __name__ == "__main__":
    if settings and settings.news_scraper_enable:
        asyncio.run(main_scraper_example())
    else:
        print("NewsScraper is not enabled in settings, or settings failed to load. Skipping example.")