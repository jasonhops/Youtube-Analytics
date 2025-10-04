"""
safetylit_scraper.py

Polite, robots-aware scraper for SafetyLit (https://www.safetylit.org/).

Usage:
    python safetylit_scraper.py

Notes:
 - Adjust START_URLS to point to the listing/archive pages you want to crawl.
 - Inspect the HTML of the pages you want to scrape and update the CSS selectors below.
 - This script checks robots.txt before crawling.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import random
import csv
import logging
from time import sleep

# --- CONFIG ---
BASE_URL = "https://www.safetylit.org/"
START_URLS = [
    # Example archive/listing pages. Change to the year/page you want to scrape.
    "https://www.safetylit.org/archive.php?year=2024",
    # add more archive/list pages as needed...
]

OUTPUT_CSV = "safetylit_articles.csv"
USER_AGENT = "MySafetyLitBot/1.0 (+https://yourorg.example/contact) " \
             "Contact: youremail@example.com"  # set a clear contact
REQUESTS_TIMEOUT = 20  # seconds
MAX_RETRIES = 3
MIN_DELAY = 1.5  # seconds (polite minimum)
MAX_DELAY = 4.0  # seconds (polite maximum)
RATE_RANDOMIZE = True

# CSS selectors - update if SafetyLit changes layout
# These are guesses — inspect the page and update accordingly.
LISTING_LINK_SELECTOR = "a[href*='citations/index.php'], a[href*='citations/viewdetails']"
# On an article page:
TITLE_SELECTOR = "h1"               # change to the real selector
AUTHORS_SELECTOR = ".authors"      # example
JOURNAL_SELECTOR = ".journal"      # example
YEAR_SELECTOR = ".year"            # example
ABSTRACT_SELECTOR = ".abstract"    # example

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Helper functions ---
def polite_sleep():
    if RATE_RANDOMIZE:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    else:
        time.sleep(MIN_DELAY)

def is_allowed_by_robots(base_url, user_agent, path):
    """
    Fetch and parse robots.txt for base_url and check permission for path.
    """
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(user_agent, path)
        logging.info("robots.txt loaded from %s — can_fetch(%s) => %s", robots_url, path, allowed)
        return allowed
    except Exception as e:
        logging.warning("Could not read robots.txt (%s). Proceeding cautiously. Error: %s", robots_url, e)
        # If robots.txt can't be read, default to conservative behavior: return False so user decides
        return False

def fetch_url(session, url):
    """
    Fetch a URL with retries and backoff.
    """
    backoff = 1.0
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logging.warning("Request failed (%s). Attempt %d/%d. Error: %s", url, attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logging.error("Max retries reached for %s", url)
                return None
            sleep(backoff)
            backoff *= 2  # exponential backoff

def extract_listing_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.select("a"):
        href = a.get("href")
        if not href:
            continue
        # convert relative links to absolute
        full = urljoin(base_url, href)
        # naive filter: include links that point to citations or detail pages
        if "citation" in full or "citations" in full or "/citations/" in full or "viewdetails" in full:
            links.add(full)
    return sorted(links)

def parse_article_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    def get_text(sel):
        el = soup.select_one(sel)
        return el.get_text(strip=True) if el else ""

    data = {
        "url": url,
        "title": get_text(TITLE_SELECTOR),
        "authors": get_text(AUTHORS_SELECTOR),
        "journal": get_text(JOURNAL_SELECTOR),
        "year": get_text(YEAR_SELECTOR),
        "abstract": get_text(ABSTRACT_SELECTOR),
    }
    return data

# --- MAIN CRAWL ---
def main():
    session = requests.Session()

    # check robots for a sample path (e.g. base url root)
    allowed = is_allowed_by_robots(BASE_URL, USER_AGENT, "/")
    if not allowed:
        logging.error("Robots.txt disallows crawling the site or couldn't be verified. Aborting.")
        print("Aborting: robots.txt disallows crawling or could not be read. Contact site owner or proceed manually.")
        return

    found_links = set()
    rows = []

    for start in START_URLS:
        logging.info("Fetching listing page: %s", start)
        r = fetch_url(session, start)
        if not r:
            continue

        links = extract_listing_links(r.text, BASE_URL)
        logging.info("Found %d candidate links on %s", len(links), start)
        for link in links:
            if link in found_links:
                continue
            # respect robots for each specific path
            parsed = urlparse(link)
            path = parsed.path + ("?" + parsed.query if parsed.query else "")
            if not is_allowed_by_robots(BASE_URL, USER_AGENT, path):
                logging.info("Skipping %s due to robots.txt disallow", link)
                continue

            logging.info("Visiting article link: %s", link)
            r2 = fetch_url(session, link)
            if not r2:
                continue

            data = parse_article_page(r2.text, link)
            # Basic sanity: skip empty titles
            if not data.get("title"):
                logging.warning("No title parsed for %s (page layout may differ).", link)
            rows.append(data)
            found_links.add(link)
            polite_sleep()

    # save to CSV
    logging.info("Saving %d records to %s", len(rows), OUTPUT_CSV)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "authors", "journal", "year", "abstract"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logging.info("Done.")

if __name__ == "__main__":
    main()

