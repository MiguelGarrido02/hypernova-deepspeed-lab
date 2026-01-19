import json
from langdetect import detect, LangDetectException, DetectorFactory

# For reproducible results
DetectorFactory.seed = 0

INPUT_FILE = r"src\scraper\data\scraped_articles.json"
OUTPUT_FILE = r"src\scraper\data\clean_scraped_articles_english.json"

# Define the footer text exactly as it appears in your scraped data.
# NOTE: If the scraper includes extra newlines or spaces, this must match exactly to work.
FOOTER_START = """About Multiverse Computing\n\nMultiverse Computing is a European company"""
FOOTER_TEXT = """About Multiverse Computing\n\nMultiverse Computing is a European company headquartered in Donostia–San Sebastián (Spain), with presence in the United States, Canada, and Europe. It is a leader in artificial intelligence software inspired by quantum technologies, focused on maximizing the performance and efficiency of large-scale AI models and infrastructures.\n\nIts compressed AI models integrate into industrial, cloud, and enterprise architectures to enable a new generation of efficient, scalable, and sovereign artificial intelligence applications.For more information:www.multiversecomputing.com"""

def filter_english_articles():
    """Filter articles to keep only those in English and clean footers."""
    
    try:
        # Load the JSON file
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles")
        
        english_articles = []
        skipped = 0
        footer_added = False  # Flag to track if we have added the footer back once
        
        for i, article in enumerate(articles):
            try:
                # Check the language of the content field
                content = article.get('content', '')
                
                if content:
                    detected_lang = detect(content)
                    
                    # Only keep articles in English
                    if detected_lang == 'en':
                        # 1. CLEANING: Cut off the article at the first mention of the footer
                        # This keeps everything BEFORE the footer, and deletes the footer + everything AFTER it.
                        if FOOTER_START in content:
                            # split returns a list [text_before, text_after]. We take [0] (text_before)
                            clean_content = content.split(FOOTER_START)[0]
                        else:
                            clean_content = content
                        
                        # 2. STRATEGY: Add the footer back ONLY to the very first article.
                        if not footer_added:
                            clean_content = clean_content.strip() + "\n\n" + FOOTER_TEXT
                            footer_added = True
                            print(f"--> Footer preserved in article: '{article.get('title', 'Unknown')}'")

                        # Update the article content
                        article['content'] = clean_content
                        
                        english_articles.append(article)
                        print(f"[{len(english_articles)}] Kept: '{article.get('title', 'Unknown')}'")
                    else:
                        print(f"Skipped article '{article.get('url', 'Unknown')}' - Language: {detected_lang}")
                        skipped += 1
                else:
                    print(f"Skipped article '{article.get('url', 'Unknown')}' - No content")
                    skipped += 1
                    
            except LangDetectException:
                print(f"Could not detect language for '{article.get('url', 'Unknown')}'")
                skipped += 1
        
        # Save the filtered articles
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(english_articles, f, indent=4, ensure_ascii=False)
        
        print(f"\nFiltered results:")
        print(f"  - English articles: {len(english_articles)}")
        print(f"  - Skipped articles: {skipped}")
        print(f"  - Saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found")
    except json.JSONDecodeError:
        print(f"Error: {INPUT_FILE} is not valid JSON")

if __name__ == "__main__":
    filter_english_articles()