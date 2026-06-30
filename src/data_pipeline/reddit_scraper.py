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
    
    def get_historical_data(self, start_date: str, end_date: str, subreddits: list = None, min_upvotes: int = 5) -> list:
        """
        Coleta posts históricos do Reddit usando a API do Arctic Shift (estável, sem rate limit agressivo).
        Fallback para PullPush caso Arctic Shift falhe.
        
        Args:
            start_date: Data inicial no formato "YYYY-MM-DD"
            end_date: Data final no formato "YYYY-MM-DD"
            subreddits: Lista de subreddits (usa TARGET_SUBREDDITS se None)
            min_upvotes: Mínimo de upvotes para filtrar posts relevantes (padrão: 5)
        """
        all_results = []
        subs = subreddits or self.TARGET_SUBREDDITS
        
        for sub in subs:
            print(f"\n📡 Coletando histórico de r/{sub} ({start_date} → {end_date})...")
            posts = self._fetch_arctic_shift(sub, start_date, end_date)
            
            # Fallback para PullPush se Arctic Shift falhar
            if posts is None:
                print(f"⚠️  Arctic Shift falhou para r/{sub}, tentando PullPush...")
                posts = self._fetch_pullpush(sub, start_date, end_date)
            
            if not posts:
                print(f"❌ Nenhum post obtido para r/{sub}")
                continue
            
            # Filtra e formata os resultados
            count_before = len(posts)
            for post in posts:
                title = post.get("title", "")
                body = post.get("selftext", "") or ""
                
                if body in ["[removed]", "[deleted]"]:
                    body = ""
                    
                full_text = f"{title}. {body}".strip()
                
                if len(full_text) < 10:
                    continue
                    
                upvotes = post.get("score", 0) or 0
                if upvotes < min_upvotes:
                    continue
                    
                permalink = post.get("full_link") or post.get("permalink", "")
                if permalink and not permalink.startswith("http"):
                    permalink = f"https://www.reddit.com{permalink}"
                    
                all_results.append({
                    "url": permalink,
                    "text": full_text[:1500],
                    "author": post.get("author", "Unknown"),
                    "created_at": datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc).isoformat(),
                    "upvotes": upvotes,
                    "comments": post.get("num_comments", 0) or 0
                })
            
            print(f"✅ r/{sub}: {len(posts)} posts brutos → {len(all_results)} relevantes (min {min_upvotes} upvotes)")
                    
        return all_results
    
    def _fetch_arctic_shift(self, subreddit: str, start_date: str, end_date: str) -> list:
        """Busca posts via Arctic Shift API (estável, sem 429)."""
        base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        all_posts = []
        current_before = end_date
        max_failures = 3
        failures = 0
        
        while True:
            params = {
                "subreddit": subreddit,
                "after": start_date,
                "before": current_before,
                "limit": 100,
                "sort": "desc",
                "sort_type": "created_utc"
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=120, 
                                       headers={"User-Agent": self.user_agent})
                
                if response.status_code != 200:
                    failures += 1
                    print(f"  ⚠️  Arctic Shift HTTP {response.status_code}")
                    if failures >= max_failures:
                        return None  # Sinaliza para usar fallback
                    time.sleep(3)
                    continue
                    
                data = response.json().get("data", [])
                
                if not data:
                    break
                    
                all_posts.extend(data)
                failures = 0
                
                # Paginação: pega o timestamp do último post
                last_ts = data[-1].get("created_utc")
                if last_ts is None:
                    break
                    
                # Converte timestamp para string ISO para o próximo request
                current_before = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                
                print(f"  📥 {len(all_posts)} posts coletados até agora...")
                time.sleep(1)  # Delay respeitoso
                
            except requests.exceptions.Timeout:
                failures += 1
                print(f"  ⏱️  Timeout na requisição ({failures}/{max_failures})")
                if failures >= max_failures:
                    return None
                time.sleep(5)
            except Exception as e:
                failures += 1
                print(f"  ❌ Erro: {e} ({failures}/{max_failures})")
                if failures >= max_failures:
                    return None
                time.sleep(3)
        
        return all_posts
    
    def _fetch_pullpush(self, subreddit: str, start_date: str, end_date: str) -> list:
        """Fallback: busca via PullPush API (mais lento, sujeito a rate limit)."""
        base_url = "https://api.pullpush.io/reddit/search/submission/"
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        
        all_posts = []
        current_end = end_ts
        failures = 0
        max_failures = 3
        
        while current_end > start_ts:
            params = {
                "subreddit": subreddit,
                "after": start_ts,
                "before": current_end,
                "size": 100,
                "sort": "desc"
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=120)
                
                if response.status_code == 429:
                    failures += 1
                    if failures >= max_failures:
                        print(f"  ⚠️  Rate limit persistente no PullPush, parando.")
                        break
                    wait = 15 * failures
                    print(f"  ⏳ Rate limit, aguardando {wait}s...")
                    time.sleep(wait)
                    continue
                elif response.status_code != 200:
                    failures += 1
                    if failures >= max_failures:
                        break
                    time.sleep(3)
                    continue
                    
                data = response.json().get("data", [])
                if not data:
                    break
                    
                all_posts.extend(data)
                failures = 0
                
                last_ts = data[-1].get("created_utc")
                if last_ts is None or last_ts >= current_end:
                    break
                current_end = last_ts - 1
                
                print(f"  📥 {len(all_posts)} posts coletados (PullPush)...")
                time.sleep(2)
                
            except Exception as e:
                failures += 1
                if failures >= max_failures:
                    print(f"  ❌ Falhas consecutivas no PullPush: {e}")
                    break
                time.sleep(5)
        
        return all_posts
