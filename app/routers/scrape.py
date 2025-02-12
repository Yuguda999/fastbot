from fastapi import APIRouter, BackgroundTasks
from app.models import ScrapeRequest
from app.utils.crawler import crawl_parallel
from app.utils.file_extraction import extract_text_from_pdf  # if needed for further processing
from app.utils.text_processing import chunk_text
from app.utils.qdrant_utils import add_embedding_to_qdrant
from app.utils.embedding import get_embedding
import asyncio, uuid, logging

router = APIRouter()
logger = logging.getLogger(__name__)

async def process_scraped_content(organization_id: str, url: str, content: str):
    from app.utils.text_processing import chunk_text
    chunks = chunk_text(content)
    for chunk in chunks:
        try:
            embedding = await get_embedding(chunk)
        except Exception as e:
            logger.error("Embedding error for %s: %s", url, e)
            continue
        doc_id = str(uuid.uuid4())
        metadata = {"url": url, "text_snippet": chunk[:200]}
        try:
            add_embedding_to_qdrant(organization_id, doc_id, embedding, metadata)
        except Exception as e:
            logger.error("Upsert error for %s: %s", url, e)

async def scrape_urls_task(organization_id: str, urls: list, max_concurrent: int = 5):
    from crawl4ai import BrowserConfig, CrawlerRunConfig, CacheMode, AsyncWebCrawler
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_url(url: str):
        async with semaphore:
            try:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session_scrape")
            except Exception as e:
                logger.error("Error crawling %s: %s", url, e)
                return

            if result.success:
                # Try different content fields
                content = (
                    getattr(result, "extracted_content", None)
                    or getattr(result, "cleaned_html", None)
                    or getattr(result, "fit_html", None)
                    or getattr(result, "markdown", None)
                    or getattr(result, "html", None)
                )

                if not content:
                    logger.error(
                        "CrawlResult does not contain content for %s. Available attributes: %s",
                        url,
                        dir(result),
                    )
                    return

                logger.info("Crawled: %s", url)
                await process_scraped_content(organization_id, url, content)
            else:
                logger.error("Failed to crawl %s: %s", url, result.error_message)


    tasks = [process_single_url(url) for url in urls if url]
    await asyncio.gather(*tasks)
    await crawler.close()

@router.post("/scrape")
async def scrape_urls_endpoint(scrape_request: ScrapeRequest):
    """
    Endpoint to start scraping a list of URLs.
    The actual crawling and Qdrant ingestion is offloaded to a Celery background task.
    """
    from app.tasks import scrape_urls_celery 
    scrape_urls_celery.delay(scrape_request.organization_id, scrape_request.urls)
    logger.info("Started scraping for org %s with %d URLs", scrape_request.organization_id, len(scrape_request.urls))
    return {
        "status": "scrape_started",
        "organization_id": scrape_request.organization_id,
        "urls_count": len(scrape_request.urls)
    }