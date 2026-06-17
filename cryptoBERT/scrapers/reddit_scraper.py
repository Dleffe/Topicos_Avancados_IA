import os
import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from dateutil import parser
import feedparser
from dotenv import load_dotenv

load_dotenv()

HOURS_WINDOW = int(os.getenv("HOURS_WINDOW", 24))

# Alvos Macro
TARGET_SUBREDDITS = ["Bitcoin", "CryptoCurrency", "CryptoMarkets", "BitcoinMarkets", "SatoshiStreetBets", "ethtrader"]

def scrape_reddit(query: str = None) -> list[dict]:

    results = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)
    
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 crypto_scraper_v1"
    
    for sub in TARGET_SUBREDDITS:
        if query:
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://www.reddit.com/r/{sub}/search.rss?q={encoded_query}&restrict_sr=on&sort=new&t=week"
        else:
            rss_url = f"https://www.reddit.com/r/{sub}/new.rss"
            
        print(f"Buscando posts via RSS no Reddit: r/{sub}...")
        feed = feedparser.parse(rss_url, agent=user_agent)
    
        for entry in feed.entries:
            try:
                pub_date = parser.parse(entry.published)
                
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                    
                if pub_date < cutoff_time:
                    continue
                    
            except Exception:
                continue
                
            author = getattr(entry, 'author', 'Unknown')
            if author.startswith('/u/'):
                author = author[3:]
                
            raw_summary = getattr(entry, 'summary', '')
            text_clean = re.sub(r'<[^>]+>', ' ', raw_summary)
            text_clean = re.sub(r'\s+', ' ', text_clean).strip()
            
            full_text = f"{entry.title}. {text_clean}"
            
            results.append({
                "url": entry.link,
                "text": full_text[:1500],
                "author": author,
                "created_at": pub_date.isoformat(),
                "upvotes": 0,
                "comments": 0
            })
        
    return results

if __name__ == "__main__":
    import json
    noticias = scrape_reddit("Bitcoin")
    if noticias:
        print(json.dumps(noticias[:2], indent=4, ensure_ascii=False))
    else:
        print("Nenhuma postagem encontrada via RSS.")
