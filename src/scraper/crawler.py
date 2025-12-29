"""
Web crawler for Robinhood support documentation.

Uses crawl4ai with Playwright for JavaScript-rendered content.
Extracts content as markdown with metadata.
"""

import asyncio
import json
import os
import re
import hashlib
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_raw_data_path


class RobinhoodCrawler:
    """Crawler for Robinhood support documentation."""
    
    BASE_URL = "https://robinhood.com/us/en/support/"
    ALLOWED_PATTERNS = [
        r"^https://robinhood\.com/us/en/support/",
    ]
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the crawler.
        
        Args:
            output_dir: Directory to save scraped content. Defaults to data/raw.
        """
        self.output_dir = output_dir or get_raw_data_path()
        self.visited_urls: set[str] = set()
        self.failed_urls: list[dict] = []
        self.crawled_count = 0
        
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        # Must match allowed patterns
        for pattern in self.ALLOWED_PATTERNS:
            if re.match(pattern, url):
                # Skip non-article URLs
                skip_patterns = [
                    r".*\.(pdf|jpg|jpeg|png|gif|svg|css|js|woff|woff2|ttf|eot|ico|mp4|mp3|webp)$",
                    r".*#.*$",  # Fragment URLs
                    r".*\?.*$",  # Query params
                    r".*/_next/.*",  # Next.js static files
                    r".*/static/.*",  # Static assets
                ]
                for skip in skip_patterns:
                    if re.match(skip, url, re.IGNORECASE):
                        return False
                return True
        return False
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove fragment and trailing slash
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    
    def _extract_metadata(self, url: str, title: str, html: str) -> dict:
        """Extract metadata from page."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        
        # Determine category from URL path
        category = "general"
        if len(path_parts) > 3:  # us/en/support/[category]/...
            category = path_parts[3]
        
        return {
            "url": url,
            "title": title,
            "category": category,
            "crawled_at": datetime.utcnow().isoformat(),
            "path": parsed.path,
        }
    
    def _generate_filename(self, url: str) -> str:
        """Generate unique filename from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        # Create hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{path}_{url_hash}.md"
        # Clean filename
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        return filename
    
    def _save_document(self, url: str, title: str, content: str, metadata: dict):
        """Save scraped document as markdown with YAML frontmatter."""
        filename = self._generate_filename(url)
        filepath = os.path.join(self.output_dir, filename)
        
        # Create YAML frontmatter
        frontmatter = "---\n"
        for key, value in metadata.items():
            frontmatter += f"{key}: {json.dumps(value)}\n"
        frontmatter += "---\n\n"
        
        # Combine frontmatter and content
        full_content = frontmatter + f"# {title}\n\n" + content
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)
        
        print(f"  Saved: {filename}")
        return filepath
    
    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML content."""
        links = []
        # Simple regex to find href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(href_pattern, html)
        
        for href in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            normalized = self._normalize_url(absolute_url)
            
            if self._is_valid_url(normalized) and normalized not in self.visited_urls:
                links.append(normalized)
        
        return list(set(links))
    
    async def crawl_page(self, crawler, url: str) -> list[str]:
        """Crawl a single page and return discovered links.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to crawl
            
        Returns:
            List of discovered URLs to crawl next
        """
        normalized_url = self._normalize_url(url)
        
        if normalized_url in self.visited_urls:
            return []
        
        self.visited_urls.add(normalized_url)
        print(f"\nCrawling [{self.crawled_count + 1}]: {normalized_url}")
        
        try:
            # Configure crawl with longer timeout and relaxed wait
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until="domcontentloaded",  # Faster than networkidle
                page_timeout=60000,  # 60 seconds
                delay_before_return_html=3.0,  # Wait 3s for JS to render
                remove_overlay_elements=True,
                exclude_external_links=True,
            )
            
            result = await crawler.arun(url=normalized_url, config=config)
            
            if not result.success:
                print(f"  Failed: {result.error_message}")
                self.failed_urls.append({"url": normalized_url, "error": result.error_message})
                return []
            
            # Extract title
            title = result.metadata.get("title", "Untitled") if result.metadata else "Untitled"
            title = title.replace(" | Robinhood", "").strip()
            
            # Get markdown content
            content = result.markdown or ""
            
            # Skip empty pages
            if len(content.strip()) < 100:
                print(f"  Skipped (empty content)")
                return []
            
            # Extract metadata
            metadata = self._extract_metadata(normalized_url, title, result.html or "")
            
            # Save document
            self._save_document(normalized_url, title, content, metadata)
            self.crawled_count += 1
            
            # Extract links for further crawling
            new_links = self._extract_links(result.html or "", normalized_url)
            print(f"  Found {len(new_links)} new links")
            
            return new_links
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            self.failed_urls.append({"url": normalized_url, "error": str(e)})
            return []
    
    async def crawl(self, max_pages: int = 500, start_url: Optional[str] = None):
        """Crawl Robinhood support documentation.
        
        Args:
            max_pages: Maximum number of pages to crawl
            start_url: Starting URL (defaults to support home)
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "crawl4ai is not installed. "
                "Please install it with: pip install crawl4ai"
            )
        
        start_url = start_url or self.BASE_URL
        
        print(f"Starting crawl from: {start_url}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max pages: {max_pages}")
        print("-" * 50)
        
        # Initialize browser config
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Queue of URLs to crawl
            queue = [start_url]
            
            while queue and self.crawled_count < max_pages:
                current_url = queue.pop(0)
                new_links = await self.crawl_page(crawler, current_url)
                
                # Add new links to queue
                for link in new_links:
                    if link not in self.visited_urls and link not in queue:
                        queue.append(link)
                
                # Small delay to be respectful
                await asyncio.sleep(0.5)
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"Crawl completed!")
        print(f"Pages crawled: {self.crawled_count}")
        print(f"Pages visited: {len(self.visited_urls)}")
        print(f"Failed URLs: {len(self.failed_urls)}")
        
        if self.failed_urls:
            print("\nFailed URLs:")
            for item in self.failed_urls[:10]:
                print(f"  - {item['url']}: {item['error']}")
        
        # Save crawl summary
        summary_path = os.path.join(self.output_dir, "_crawl_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "crawled_count": self.crawled_count,
                "visited_urls": list(self.visited_urls),
                "failed_urls": self.failed_urls,
                "completed_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        return self.crawled_count


async def main():
    """Main entry point for crawler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawl Robinhood support documentation")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to crawl")
    parser.add_argument("--start-url", type=str, default=None, help="Starting URL")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--test-url", type=str, default=None, help="Test single URL")
    
    args = parser.parse_args()
    
    crawler = RobinhoodCrawler(output_dir=args.output_dir)
    
    if args.test_url:
        # Test mode: crawl single page
        print(f"Testing single URL: {args.test_url}")
        browser_config = BrowserConfig(headless=True, verbose=False)
        async with AsyncWebCrawler(config=browser_config) as web_crawler:
            await crawler.crawl_page(web_crawler, args.test_url)
    else:
        # Full crawl
        await crawler.crawl(
            max_pages=args.max_pages,
            start_url=args.start_url
        )


if __name__ == "__main__":
    asyncio.run(main())
