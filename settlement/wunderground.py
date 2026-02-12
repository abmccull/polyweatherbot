"""Selenium scraper for Weather Underground history page (JS-rendered)."""

from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from threading import Lock

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from utils.logging import get_logger

log = get_logger("wunderground")

WU_BASE = "https://www.wunderground.com/history/daily"


@dataclass
class WUDailyHigh:
    """Scraped daily high from Weather Underground."""

    temp_c: int  # WU displays whole degrees
    raw_text: str
    scraped_at: datetime
    url: str


class WundergroundScraper:
    """Scrapes Weather Underground history pages for daily high temperatures.

    WU pages are JS-rendered (Angular), so we use Selenium with headless Chrome.
    Results are cached for `cache_ttl` seconds to avoid excessive scraping.
    """

    def __init__(self, cache_ttl: int = 120) -> None:
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[WUDailyHigh, float]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wu_scraper")
        self._driver: webdriver.Chrome | None = None
        self._driver_lock = Lock()

    def _get_driver(self) -> webdriver.Chrome:
        """Get or create the Selenium Chrome driver."""
        if self._driver is None:
            with self._driver_lock:
                if self._driver is None:
                    options = Options()
                    options.add_argument("--headless=new")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--disable-gpu")
                    options.add_argument("--window-size=1920,1080")
                    options.add_argument(
                        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    # Prefer system chromedriver if available
                    import shutil
                    system_driver = shutil.which("chromedriver")
                    if system_driver:
                        service = Service(system_driver)
                    else:
                        service = Service(ChromeDriverManager().install())
                    self._driver = webdriver.Chrome(service=service, options=options)
        return self._driver

    def _build_url(self, wu_path: str, for_date: date) -> str:
        """Build WU history URL.

        Pattern: https://www.wunderground.com/history/daily/{wu_path}/date/{YYYY-M-D}
        Note: WU uses non-zero-padded month and day.
        """
        return f"{WU_BASE}/{wu_path}/date/{for_date.year}-{for_date.month}-{for_date.day}"

    def _scrape_sync(self, wu_path: str, for_date: date) -> WUDailyHigh | None:
        """Synchronous scraping — runs in thread pool."""
        url = self._build_url(wu_path, for_date)
        cache_key = f"{wu_path}:{for_date}"

        # Check cache
        with self._lock:
            if cache_key in self._cache:
                cached, ts = self._cache[cache_key]
                if time.time() - ts < self._cache_ttl:
                    return cached

        try:
            driver = self._get_driver()
            driver.get(url)

            # Wait for the observation table to render
            wait = WebDriverWait(driver, 20)
            table = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.observation-table"))
            )

            # Find the daily summary rows — look for "Max" or "High" row
            rows = table.find_elements(By.TAG_NAME, "tr")
            high_temp = None
            raw_text = ""

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                header = row.find_elements(By.TAG_NAME, "th")

                row_label = ""
                if header:
                    row_label = header[0].text.strip().lower()
                elif cells:
                    row_label = cells[0].text.strip().lower()

                if "max" in row_label or "high" in row_label:
                    # Temperature is typically in the first or second data cell
                    for cell in cells:
                        text = cell.text.strip()
                        # Look for temperature pattern (number with optional °)
                        temp_match = re.search(r'(-?\d+)\s*°?\s*[CF]?', text)
                        if temp_match:
                            raw_text = text
                            high_temp = int(temp_match.group(1))
                            break
                    if high_temp is not None:
                        break

            if high_temp is None:
                # Alternative: try to find daily summary section
                try:
                    summary_elements = driver.find_elements(
                        By.CSS_SELECTOR, ".daily-summary .summary-value, .observation-table td"
                    )
                    for el in summary_elements:
                        text = el.text.strip()
                        temp_match = re.search(r'(-?\d+)\s*°', text)
                        if temp_match:
                            high_temp = int(temp_match.group(1))
                            raw_text = text
                            break
                except Exception:
                    pass

            if high_temp is None:
                log.warning("wu_high_not_found", url=url)
                return None

            result = WUDailyHigh(
                temp_c=high_temp,
                raw_text=raw_text,
                scraped_at=datetime.utcnow(),
                url=url,
            )

            # Cache result
            with self._lock:
                self._cache[cache_key] = (result, time.time())

            log.info("wu_scraped", wu_path=wu_path, date=str(for_date), high_c=high_temp)
            return result

        except Exception as e:
            log.error("wu_scrape_failed", url=url, error=str(e))
            return None

    async def get_daily_high(self, wu_path: str, for_date: date) -> WUDailyHigh | None:
        """Async wrapper: scrape WU daily high in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._scrape_sync, wu_path, for_date)

    def close(self) -> None:
        """Close Selenium driver and executor."""
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
        self._executor.shutdown(wait=False)
