import os
import urllib.parse
import time
from datetime import datetime, timezone, timedelta
from dateutil import parser, relativedelta
import feedparser
from dotenv import load_dotenv

load_dotenv()

class GoogleNewsScraper:
    
    CRYPTO_RSS_FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cryptoslate.com/feed/",
        "https://bitcoinmagazine.com/feed"
    ]
    
    def __init__(self):
        self.hours_window = int(os.getenv("HOURS_WINDOW", 24))
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    def get_current_data(self, query: str) -> list:
        encoded_query = urllib.parse.quote(query)
        gnews_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        all_feeds = [gnews_url] + self.CRYPTO_RSS_FEEDS
        results = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.hours_window)
        
        for rss_url in all_feeds:
            print(f"Buscando RSS em: {rss_url.split('/')[2]}")
            feed = feedparser.parse(rss_url, agent=self.user_agent)
            
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
    
    def get_historical_data(self, query: str, start_date: str, end_date: str) -> list:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        results = []
        seen_links = set()
        
        current_dt = start_dt
        while current_dt < end_dt:
            next_dt = current_dt + relativedelta.relativedelta(weeks=1)
            if next_dt > end_dt:
                next_dt = end_dt
                
            str_after = current_dt.strftime("%Y-%m-%d")
            str_before = next_dt.strftime("%Y-%m-%d")
            
            advanced_query = f"{query} after:{str_after} before:{str_before}"
            encoded_query = urllib.parse.quote(advanced_query)
            
            gnews_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            print(f"Buscando período: {str_after} a {str_before}")
            feed = feedparser.parse(gnews_url, agent=self.user_agent)
            
            for entry in feed.entries:
                link = getattr(entry, 'link', '')
                if link in seen_links:
                    continue
                    
                seen_links.add(link)
                
                try:
                    pub_date = parser.parse(entry.published)
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                    
                source_name = "News Outlet"
                if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                    source_name = entry.source.title
                elif hasattr(feed, 'feed') and hasattr(feed.feed, 'title'):
                    source_name = feed.feed.title
                    
                results.append({
                    "title": getattr(entry, 'title', ''),
                    "desc": getattr(entry, 'description', ''),
                    "link": link,
                    "date": pub_date.isoformat(),
                    "source": source_name
                })
                
            current_dt = next_dt
            time.sleep(2)
            
        return results
