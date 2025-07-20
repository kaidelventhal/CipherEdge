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