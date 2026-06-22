import os
import re
import urllib.parse
import time
from datetime import datetime, timezone, timedelta
from dateutil import parser
import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

class RedditScraper:
    
    TARGET_SUBREDDITS = ["Bitcoin", "CryptoCurrency", "CryptoMarkets", "BitcoinMarkets", "SatoshiStreetBets", "ethtrader"]
    
    def __init__(self):
        self.hours_window = int(os.getenv("HOURS_WINDOW", 24))
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 crypto_scraper_v1"
    
    def get_current_data(self, query: str = None, subreddits: list = None) -> list:
        results = []
        subs = subreddits or self.TARGET_SUBREDDITS
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.hours_window)
        
        for sub in subs:
            if query:
                encoded_query = urllib.parse.quote(query)
                rss_url = f"https://www.reddit.com/r/{sub}/search.rss?q={encoded_query}&restrict_sr=on&sort=new&t=week"
            else:
                rss_url = f"https://www.reddit.com/r/{sub}/new.rss"
                
            print(f"Buscando posts via RSS no Reddit: r/{sub}...")
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
    
    def get_historical_data(self, start_date: str, end_date: str, subreddits: list = None) -> list:
        base_url = "https://api.pullpush.io/reddit/search/submission/"
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        
        all_results = []
        subs = subreddits or self.TARGET_SUBREDDITS
        
        for sub in subs:
            current_end_ts = end_ts
            
            while current_end_ts > start_ts:
                params = {
                    "subreddit": sub,
                    "after": start_ts,
                    "before": current_end_ts,
                    "size": 100,
                    "sort": "desc"
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=10)
                    
                    if response.status_code == 429:
                        time.sleep(10)
                        continue
                    elif response.status_code != 200:
                        time.sleep(2)
                        continue
                        
                    data = response.json().get("data", [])
                    
                    if not data:
                        break
                        
                    for post in data:
                        title = post.get("title", "")
                        body = post.get("selftext", "")
                        
                        if body in ["[removed]", "[deleted]"]:
                            body = ""
                            
                        full_text = f"{title}. {body}".strip()
                        
                        if len(full_text) < 10:
                            continue
                            
                        all_results.append({
                            "url": post.get("full_link", ""),
                            "text": full_text[:1500],
                            "author": post.get("author", "Unknown"),
                            "created_at": datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc).isoformat(),
                            "upvotes": post.get("score", 0),
                            "comments": post.get("num_comments", 0)
                        })
                        
                    current_end_ts = data[-1]["created_utc"]
                    time.sleep(1.5)
                    
                except Exception:
                    time.sleep(5)
                    
        return all_results
