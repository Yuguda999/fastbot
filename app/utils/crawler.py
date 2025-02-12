import asyncio
import logging
from typing import List, Set
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from app.utils.url_utils import is_valid_url, is_internal_url, normalize_url
from urllib.parse import urljoin, urlparse, urlunparse

logger = logging.getLogger(__name__)

async def worker(queue: asyncio.Queue, visited: set, base_url: str, context, max_pages: int):
    while True:
        try:
            current_url = await queue.get()
        except asyncio.CancelledError:
            break

        norm_current = normalize_url(current_url)
        if norm_current in visited or len(visited) >= max_pages:
            queue.task_done()
            continue

        logger.info("Visiting: %s", current_url)
        try:
            page = await context.new_page()
            await page.goto(current_url, timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=15000)
            content = await page.content()
            await page.close()
        except PlaywrightTimeoutError as e:
            logger.error("Timeout loading %s: %s", current_url, e)
            try:
                await page.close()
            except Exception:
                pass
            queue.task_done()
            continue
        except Exception as e:
            logger.error("Error loading %s: %s", current_url, e)
            try:
                await page.close()
            except Exception:
                pass
            queue.task_done()
            continue

        visited.add(norm_current)
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as e:
            logger.error("Error parsing HTML from %s: %s", current_url, e)
            queue.task_done()
            continue

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(current_url, href)
            if is_valid_url(full_url) and is_internal_url(full_url, base_url):
                norm_full_url = normalize_url(full_url)
                if norm_full_url not in visited:
                    await queue.put(full_url)
        queue.task_done()

async def crawl_site_js_fast(start_url: str, max_pages: int = 100, concurrent_workers: int = 20) -> Set[str]:
    visited = set()
    queue = asyncio.Queue()
    await queue.put(start_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        tasks = [
            asyncio.create_task(worker(queue, visited, start_url, context, max_pages))
            for _ in range(concurrent_workers)
        ]
        
        await queue.join()
        
        for task in tasks:
            task.cancel()
        await browser.close()
    return visited

# An alternative parallel crawler using AsyncWebCrawler (from crawl4ai)
from crawl4ai import BrowserConfig, CrawlerRunConfig, CacheMode, AsyncWebCrawler

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_url(url: str):
        async with semaphore:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id="session1"
            )
            if result.success:
                logger.info("Successfully crawled: %s", url)
            else:
                logger.error("Failed: %s - %s", url, result.error_message)

    await asyncio.gather(*[process_url(url) for url in urls])
    await crawler.close()
