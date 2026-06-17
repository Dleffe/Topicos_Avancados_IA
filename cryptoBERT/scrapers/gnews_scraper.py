import os
import urllib.parse
from datetime import datetime, timezone, timedelta
from dateutil import parser
import feedparser
from dotenv import load_dotenv

load_dotenv()

HOURS_WINDOW = int(os.getenv("HOURS_WINDOW", 24))

# Feeds Diretos de Cripto
CRYPTO_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
    "https://bitcoinmagazine.com/feed"
]

def scrape_gnews(query: str) -> list[dict]:

    encoded_query = urllib.parse.quote(query)
    gnews_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    all_feeds = [gnews_url] + CRYPTO_RSS_FEEDS
    
    results = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)
    
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    for rss_url in all_feeds:
        print(f"Buscando RSS em: {rss_url.split('/')[2]}")
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
                
            if rss_url != gnews_url and query.lower() not in entry.title.lower():
                continue
                
            source_name = "News Outlet"
            if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                source_name = entry.source.title
            elif hasattr(feed, 'feed') and hasattr(feed.feed, 'title'):
                source_name = feed.feed.title
                
            results.append({
                "title": entry.title,
                "desc": getattr(entry, 'description', ''),
                "link": entry.link,
                "date": pub_date.isoformat(),
                "source": source_name
            })
            
    return results

if __name__ == "__main__":
    import json
    noticias = scrape_gnews("Bitcoin")
    if noticias:
        print(json.dumps(noticias[:2], indent=4, ensure_ascii=False))
    else:
        print("Nenhuma notícia encontrada.")