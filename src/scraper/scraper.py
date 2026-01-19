import time
from bs4 import BeautifulSoup
import json
import requests


# --- CONFIGURATION ---
BASE_URL = "https://multiversecomputing.com"
RESOURCES_URL = "https://multiversecomputing.com/resources"
OUTPUT_FILE = "scraped_articles.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# The link class we confirmed in the previous step
LINK_CLASS = "grid grid-cols-1 gap-5 transition-opacity duration-300 visited:opacity-100 hover:opacity-80"

def get_all_article_urls():
    """Fetch all article URLs by finding the last page where count stops increasing."""
    
    page = 1
    previous_count = 0
    current_count = 0
    article_urls = []
    
    while True:
        print(f"Fetching page {page}...")
        try:
            response = requests.get(f"{RESOURCES_URL}?page={page}&language=English", headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links using the class we verified earlier
            links = soup.find_all('a', class_=LINK_CLASS)
            current_count = len(links)
            
            print(f"Page {page}: Found {current_count} total articles")
            
            # If count doesn't increase, we've reached the last page
            if current_count == previous_count:
                print(f"Article count stopped increasing. Last page is {page - 1}")
                # Fetch the previous page (last page with new articles)
                page -= 1
                response = requests.get(f"{RESOURCES_URL}?page={page}&language=English", headers=HEADERS, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', class_=LINK_CLASS)
                break
            
            previous_count = current_count
            page += 1
            time.sleep(0.5)  # Polite delay between requests
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            return []
    
    # Extract all URLs from the last page
    for link in links:
        href = link.get('href')
        if href:
            # Normalize URL
            if href.startswith('/'):
                full_url = BASE_URL + href
            else:
                full_url = href
            article_urls.append(full_url)
    
    print(f"Scraped {len(article_urls)} article URLs from the last page")
    return article_urls



def scrape_single_article(url):
    """Step 2: Scrape details from a single URL using Requests (Faster)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- 1. EXTRACT TITLE ---
        # <h1 class="text-xl font-medium">
        title_tag = soup.find('h1', class_="text-xl font-medium")
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # --- 2. EXTRACT DATE ---
        # <div class="text-sm text-gray-8">
        date_tag = soup.find('div', class_="text-sm text-gray-8")
        date = date_tag.get_text(strip=True) if date_tag else ""
        
        # --- 3. EXTRACT CONTENT ---
        # <article class=""> ... <p> ... </p> </article>
        content = ""
        article_tag = soup.find('article')
        if article_tag:
            # Get text from all paragraphs, joined by newlines
            paragraphs = article_tag.find_all('p')
            content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
        
        return {
            "url": url,
            "title": title,
            "date": date,
            "content": content
        }

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def main():
    # 1. Get List
    urls = get_all_article_urls()
    print(f"Found {len(urls)} articles. Starting detailed scrape...")
    
    final_data = []
    
    # 2. Loop through every URL
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Scraping: {url}")
        
        data = scrape_single_article(url)
        
        if data:
            # Basic validation: if title is empty, something might have changed
            if not data['title']:
                print(f"Warning: No title found for {url}")
                
            final_data.append(data)
        
        # Polite delay to avoid getting banned
        time.sleep(0.5)

    # 3. Save to JSON
    print(f"--- Saving data to {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        
    print("Done!")

    print("-"*40)

    #View example output
    if final_data:
        print("Example scraped article:")
        print(json.dumps(final_data[0], indent=4, ensure_ascii=False))



if __name__ == "__main__":
    main()