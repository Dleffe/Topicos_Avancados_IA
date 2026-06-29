import os
import re
import urllib.parse
import time
from datetime import datetime, timezone, timedelta
from dateutil import parser
import feedparser
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from tqdm import tqdm
import praw

load_dotenv()

class RedditScraper:
    
    TARGET_SUBREDDITS = ["Bitcoin", "CryptoCurrency", "CryptoMarkets", "BitcoinMarkets", "SatoshiStreetBets", "ethtrader"]
    
    def __init__(self):
        self.hours_window = int(os.getenv("HOURS_WINDOW", 24))
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 crypto_scraper_v1"
        
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
        
        if self.client_id and self.client_secret and self.reddit_user_agent:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.reddit_user_agent
            )
        else:
            self.reddit = None
            print("Credenciais da API do Reddit não encontradas. A coleta de dados históricos será desabilitada.")

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
        if not self.reddit:
            print("Coleta de dados históricos do Reddit desabilitada devido à falta de credenciais da API.")
            return []

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        
        all_results = []
        subs = subreddits or self.TARGET_SUBREDDITS
        
        for sub_name in tqdm(subs, desc="Processing Subreddits"):
            subreddit = self.reddit.subreddit(sub_name)
            query = f"timestamp:{start_ts}..{end_ts}"
            
            try:
                for post in tqdm(subreddit.search(query, sort="new", syntax="cloudsearch"), desc=f"r/{sub_name}", leave=False):
                    title = post.title
                    body = post.selftext
                    
                    if body in ["[removed]", "[deleted]"]:
                        body = ""
                        
                    full_text = f"{title}. {body}".strip()
                    
                    if len(full_text) < 10:
                        continue
                        
                    all_results.append({
                        "url": f"https://www.reddit.com{post.permalink}",
                        "text": full_text[:1500],
                        "author": post.author.name if post.author else "Unknown",
                        "created_at": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                        "upvotes": post.score,
                        "comments": post.num_comments
                    })
            except Exception as e:
                print(f"Erro ao buscar dados no subreddit r/{sub_name}: {e}")

        return all_results
