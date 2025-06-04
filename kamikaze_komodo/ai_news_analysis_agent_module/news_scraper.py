# kamikaze_komodo/ai_news_analysis_agent_module/news_scraper.py
import asyncio
import feedparser
import newspaper
import httpx # Using httpx for async requests, aiohttp is also good
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup # For more complex parsing if needed

from kamikaze_komodo.core.models import NewsArticle
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

class NewsScraper:
    """
    Scrapes news from specified sources (RSS feeds, websites).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        self.rss_feeds = self.config.get("rss_feeds", []) # List of dicts like {'name': 'CoinTelegraph', 'url': '...'}
        self.websites = self.config.get("websites", []) # List of dicts for direct scraping {'name': 'CoinDesk', 'url': '...'}
        
        # Example sources from plan
        # self.rss_feeds = [
        #     {"name": "CoinTelegraph RSS", "url": "https://cointelegraph.com/rss"},
        #     {"name": "SeekingAlpha Market Currents", "url": "https://seekingalpha.com/market_currents.xml"}
        # ]
        # self.websites_to_scrape = [ # For Newspaper3k or custom BS4
        #     {"name": "CoinDesk", "url": "https://www.coindesk.com/"},
        #     {"name": "CoinTelegraph", "url": "https://cointelegraph.com/"} # Besides RSS
        # ]
        logger.info(f"NewsScraper initialized. RSS feeds: {len(self.rss_feeds)}, Websites: {len(self.websites)}")

    async def _fetch_url_content(self, url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 KamikazeKomodoBot/1.0'}
                response = await client.get(url, headers=headers)
                response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                return response.text
        except httpx.RequestError as e:
            logger.error(f"Error fetching URL {url}: {e}")
        except Exception as e_gen:
            logger.error(f"Generic error fetching URL {url}: {e_gen}")
        return None

    async def scrape_rss_feed(self, feed_name: str, feed_url: str, limit: int = 10) -> List[NewsArticle]:
        """Scrapes a single RSS feed."""
        articles: List[NewsArticle] = []
        logger.info(f"Scraping RSS feed: {feed_name} from {feed_url}")
        try:
            # feedparser is synchronous, run in executor for async context
            loop = asyncio.get_event_loop()
            parsed_feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
            
            if parsed_feed.bozo: # If non-None, indicates an error during parsing
                logger.warning(f"Error parsing RSS feed {feed_name} ({feed_url}): {parsed_feed.bozo_exception}")
                # Fallback or retry with httpx if feedparser fails badly due to network etc.
                # content = await self._fetch_url_content(feed_url)
                # if content:
                #     parsed_feed = await loop.run_in_executor(None, feedparser.parse, content)
                # else:
                #     return articles # Could not fetch content

            for entry in parsed_feed.entries[:limit]:
                title = entry.get("title")
                link = entry.get("link")
                published_time = entry.get("published_parsed")
                
                pub_date = None
                if published_time:
                    try:
                        pub_date = datetime(*published_time[:6], tzinfo=timezone.utc)
                    except Exception as e_date:
                        logger.warning(f"Could not parse date for article '{title}': {published_time}, error: {e_date}")

                article_id = link # Use URL as a unique ID
                
                # Use newspaper3k to get full content if desired, can be slow for many articles
                # For now, let's take summary if available, or description
                content_summary = entry.get("summary") or entry.get("description")

                articles.append(NewsArticle(
                    id=article_id,
                    url=link,
                    title=title,
                    publication_date=pub_date,
                    retrieval_date=datetime.now(timezone.utc),
                    source=feed_name,
                    content=None, # Full content fetch can be added here or later
                    summary=content_summary
                ))
            logger.info(f"Found {len(articles)} articles from RSS feed: {feed_name}")
        except Exception as e:
            logger.error(f"Failed to scrape RSS feed {feed_name} ({feed_url}): {e}", exc_info=True)
        return articles

    async def scrape_website_with_newspaper(self, site_name: str, site_url: str, limit_articles: int = 5) -> List[NewsArticle]:
        """Scrapes a website using Newspaper3k."""
        articles_data: List[NewsArticle] = []
        logger.info(f"Scraping website: {site_name} ({site_url}) with Newspaper3k")
        try:
            # Newspaper3k is synchronous, consider running parts in executor or using an async alternative if performance is critical for many sites.
            # For a few primary sites, this might be acceptable.
            loop = asyncio.get_event_loop()
            
            paper = await loop.run_in_executor(None, newspaper.build, site_url, {'memoize_articles': False, 'request_timeout': 15, 'browser_user_agent': 'Mozilla/5.0 KamikazeKomodoBot/1.0'})
            
            count = 0
            for article_raw in paper.articles:
                if count >= limit_articles:
                    break
                try:
                    await loop.run_in_executor(None, article_raw.download)
                    if not article_raw.is_downloaded:
                        logger.warning(f"Failed to download article: {article_raw.url} from {site_name}")
                        continue
                        
                    await loop.run_in_executor(None, article_raw.parse)
                    
                    title = article_raw.title
                    url = article_raw.url
                    content = article_raw.text
                    pub_date_dt = article_raw.publish_date
                    
                    # Ensure pub_date_dt is timezone-aware (UTC)
                    if pub_date_dt and pub_date_dt.tzinfo is None:
                        pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc) # Assume UTC if naive

                    articles_data.append(NewsArticle(
                        id=url,
                        url=url,
                        title=title,
                        publication_date=pub_date_dt,
                        retrieval_date=datetime.now(timezone.utc),
                        source=site_name,
                        content=content,
                        summary=None # Newspaper3k can generate summary: article_raw.summary (requires NLP)
                    ))
                    count += 1
                    logger.debug(f"Successfully scraped: {url} from {site_name}")
                except Exception as e_article:
                    logger.warning(f"Error processing article {article_raw.url} from {site_name}: {e_article}")
            logger.info(f"Scraped {len(articles_data)} articles from {site_name} using Newspaper3k.")
        except Exception as e:
            logger.error(f"Failed to scrape website {site_name} ({site_url}) with Newspaper3k: {e}", exc_info=True)
        return articles_data


    async def scrape_all(self, limit_per_source: int = 10) -> List[NewsArticle]:
        """Scrapes all configured RSS feeds and websites."""
        all_articles: List[NewsArticle] = []
        
        # Scrape RSS Feeds
        rss_tasks = []
        for feed_info in self.rss_feeds:
            rss_tasks.append(self.scrape_rss_feed(feed_info['name'], feed_info['url'], limit=limit_per_source))
        
        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
        for result in rss_results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"RSS scraping task failed: {result}")

        # Scrape Websites (e.g., using Newspaper3k)
        website_tasks = []
        for site_info in self.websites: # Assuming 'websites' config like RSS for now
            website_tasks.append(self.scrape_website_with_newspaper(site_info['name'], site_info['url'], limit_articles=limit_per_source))
        
        website_results = await asyncio.gather(*website_tasks, return_exceptions=True)
        for result in website_results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Website scraping task failed: {result}")
        
        # Deduplicate articles by URL (ID)
        unique_articles_dict = {article.id: article for article in all_articles}
        unique_articles_list = list(unique_articles_dict.values())
        
        logger.info(f"Total unique articles scraped from all sources: {len(unique_articles_list)}")
        return unique_articles_list

# Example Usage (for testing)
async def main_scraper_example(settings_instance):
    """ Example of using the NewsScraper """
    scraper_config = {
        "rss_feeds": [
            {"name": "CoinTelegraph RSS", "url": settings_instance.rss_feed_cointelegraph},
            {"name": "SeekingAlpha Market Currents", "url": settings_instance.rss_feed_seeking_alpha}
        ] if settings_instance else [],
        "websites": [ # Example: add website scraping tasks here if desired for Newspaper3k
             {"name": "CoinDesk", "url": "https://www.coindesk.com/consensus-magazine/"}, # Specific section
             # {"name": "CoinTelegraph", "url": "https://cointelegraph.com/"} # Main site can be heavy
        ] if settings_instance else []
    }
    scraper = NewsScraper(config=scraper_config)
    
    # Scrape specific RSS feed
    # ct_articles = await scraper.scrape_rss_feed("CoinTelegraph RSS", scraper_config['rss_feeds'][0]['url'])
    # for article in ct_articles[:2]:
    #     print(f"Title: {article.title}, URL: {article.url}, Source: {article.source}, Date: {article.publication_date}")

    # Scrape specific website
    # coindesk_articles = await scraper.scrape_website_with_newspaper("CoinDesk", scraper_config['websites'][0]['url'], limit_articles=2)
    # for article in coindesk_articles:
    #      print(f"Site: {article.source}, Title: {article.title}, URL: {article.url}, Date: {article.publication_date}, Content len: {len(article.content) if article.content else 0}")


    # Scrape all configured sources
    all_scraped_articles = await scraper.scrape_all(limit_per_source=3) # Limit to 3 articles per source for example
    logger.info(f"--- All Scraped Articles ({len(all_scraped_articles)}) ---")
    for i, article in enumerate(all_scraped_articles):
        logger.info(f"{i+1}. Source: {article.source}, Title: {article.title}, Date: {article.publication_date}, URL: {article.url}")
        if article.summary:
            logger.info(f"   Summary: {article.summary[:100]}...")
        # if article.content: # Content can be very long
        #    logger.info(f"   Content: {article.content[:100]}...")


# if __name__ == "__main__":
#     # This is a placeholder for running the example.
#     # You'd need to ensure 'settings' is loaded or pass config directly.
#     # from kamikaze_komodo.config.settings import settings as app_settings
#     # if app_settings:
#     #    asyncio.run(main_scraper_example(app_settings))
#     # else:
#     #    print("App settings not loaded, cannot run scraper example.")
#     pass