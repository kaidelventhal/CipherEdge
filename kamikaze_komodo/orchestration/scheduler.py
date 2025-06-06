# kamikaze_komodo/orchestration/scheduler.py

from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
import os

logger = get_logger(__name__)

class TaskScheduler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TaskScheduler, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, db_path: Optional[str] = "logs/scheduler_jobs.sqlite"):
        if hasattr(self, '_initialized') and self._initialized: # Ensure __init__ runs only once for singleton
            return
        
        if not settings:
            logger.critical("Settings not loaded. TaskScheduler cannot be initialized.")
            raise ValueError("Settings not loaded.")

        self.db_path = db_path
        if self.db_path and not os.path.isabs(self.db_path):
             # Get project root based on this file's location: kamikaze_komodo/orchestration/scheduler.py
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(project_root, self.db_path)
        
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory for scheduler database: {db_dir}")

        jobstores = {
            'default': SQLAlchemyJobStore(url=f'sqlite:///{self.db_path}')
        }
        executors = {
            'default': ThreadPoolExecutor(10), # For I/O bound tasks
            'processpool': ProcessPoolExecutor(3) # For CPU bound tasks
        }
        job_defaults = {
            'coalesce': False, # Run missed jobs if scheduler was down (be careful with this)
            'max_instances': 3 # Max parallel instances of the same job
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC' # Explicitly set timezone
        )
        self._initialized = True
        logger.info(f"TaskScheduler initialized with SQLite job store at: {self.db_path}")

    def start(self):
        if not self.scheduler.running:
            try:
                self.scheduler.start()
                logger.info("APScheduler started.")
            except Exception as e:
                logger.error(f"Failed to start APScheduler: {e}", exc_info=True)
        else:
            logger.info("APScheduler is already running.")

    def shutdown(self, wait: bool = True):
        if self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=wait)
                logger.info("APScheduler shut down.")
            except Exception as e:
                logger.error(f"Error shutting down APScheduler: {e}", exc_info=True)

    def add_job(self, func, trigger: str = 'interval', **kwargs):
        """
        Adds a job to the scheduler.
        Args:
            func: The function to execute.
            trigger: The trigger type (e.g., 'interval', 'cron', 'date').
            **kwargs: Arguments for the trigger and job (e.g., minutes=1, id='my_job').
        """
        try:
            job = self.scheduler.add_job(func, trigger, **kwargs)
            logger.info(f"Job '{kwargs.get('id', func.__name__)}' added with trigger: {trigger}, params: {kwargs}")
            return job
        except Exception as e:
            logger.error(f"Failed to add job '{kwargs.get('id', func.__name__)}': {e}", exc_info=True)
            return None

    def remove_job(self, job_id: str):
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Job '{job_id}' removed.")
        except Exception as e: # Specific exception: JobLookupError
            logger.warning(f"Failed to remove job '{job_id}': {e}")

# --- Example Scheduled Tasks (Conceptual for Phase 6 "Begin Integration") ---
async def example_data_polling_task():
    logger.info("Scheduler: Running example_data_polling_task...")
    # In a real scenario, this would call a method in DataFetcher or a dedicated data polling module.
    # from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
    # fetcher = DataFetcher()
    # await fetcher.fetch_latest_data_for_active_symbols() # Fictional method
    # await fetcher.close()
    await asyncio.sleep(2) # Simulate work
    logger.info("Scheduler: example_data_polling_task finished.")

async def example_news_scraping_task():
    logger.info("Scheduler: Running example_news_scraping_task...")
    # from kamikaze_komodo.ai_news_analysis_agent_module.news_scraper import NewsScraper
    # scraper = NewsScraper()
    # articles = await scraper.scrape_all(limit_per_source=5, since_hours_rss=12)
    # if articles:
    #     logger.info(f"Scheduler: Scraped {len(articles)} news articles.")
        # Further processing: store, analyze sentiment, etc.
    await asyncio.sleep(5) # Simulate work
    logger.info("Scheduler: example_news_scraping_task finished.")

async def example_model_retraining_check_task():
    logger.info("Scheduler: Running example_model_retraining_check_task...")
    # This task would check conditions for retraining ML models
    # e.g., time since last training, performance degradation, new data volume
    # from kamikaze_komodo.ml_models.training_pipelines.lightgbm_pipeline import LightGBMTrainingPipeline
    # if conditions_met_for_retraining("LightGBM_BTCUSD_1h"):
    #     pipeline = LightGBMTrainingPipeline(symbol="BTC/USD", timeframe="1h")
    #     await pipeline.run_training()
    await asyncio.sleep(3)
    logger.info("Scheduler: example_model_retraining_check_task finished.")


async def main_scheduler_example():
    """Demonstrates basic scheduler setup and job addition."""
    scheduler_manager = TaskScheduler()

    # Add example jobs (these won't run unless scheduler is started and loop runs)
    scheduler_manager.add_job(example_data_polling_task, 'interval', minutes=15, id='data_poll_main')
    scheduler_manager.add_job(example_news_scraping_task, 'cron', hour='*/2', id='news_scrape_bi_hourly') # Every 2 hours
    scheduler_manager.add_job(example_model_retraining_check_task, 'cron', day_of_week='sun', hour='3', minute='0', id='weekly_retrain_check')

    try:
        scheduler_manager.start()
        # Keep the main thread alive to allow scheduler to run, or integrate into main application loop
        # For this example, we'll just let it run for a short period.
        # In a real app, asyncio.get_event_loop().run_forever() or similar would be used.
        if settings and settings.log_level.upper() == "DEBUG": # Only run for a bit in debug
            logger.info("Scheduler example running for 30 seconds (DEBUG mode)...")
            await asyncio.sleep(30)
        else:
            logger.info("Scheduler example configured. In a full app, it would run continuously.")
            # For non-debug, don't block indefinitely here in a simple example.
            # Real app would have its own main loop.
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler example interrupted.")
    finally:
        scheduler_manager.shutdown()

if __name__ == "__main__":
    import asyncio
    # This example shows how to set up the scheduler.
    # It's best integrated into the main application's async loop (e.g., in main.py).
    # Run with: python -m kamikaze_komodo.orchestration.scheduler
    
    # To see jobs persist and get reloaded, run, stop, then run again.
    # Check the logs/scheduler_jobs.sqlite file.
    
    # Note: Running this standalone will schedule jobs. If you run it multiple times
    # without clearing the scheduler_jobs.sqlite, it might try to add duplicate jobs
    # if the `replace_existing=True` option is not used in add_job and IDs are the same.
    # The current setup will log errors for duplicate job IDs if they are not replaced.
    asyncio.run(main_scheduler_example())