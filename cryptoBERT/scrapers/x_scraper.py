#Código não funcionando pois a biblioteca não está atualizada

import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from twikit import Client, TooManyRequests

load_dotenv()

HOURS_WINDOW = int(os.getenv("HOURS_WINDOW", 24))
MIN_LIKES = int(os.getenv("MIN_LIKES", 100))
MIN_RETWEETS = int(os.getenv("MIN_RETWEETS", 10))
X_USERNAME = os.getenv("X_USERNAME")
X_EMAIL = os.getenv("X_EMAIL")
X_PASSWORD = os.getenv("X_PASSWORD")
X_COOKIES_PATH = os.getenv("X_COOKIES_PATH", "cookies.json")

async def scrape_x(query: str, max_results: int = 50) -> list[dict]:
    client = Client('en-US')
    
    try:
        await client.login(
            auth_info_1=X_USERNAME,
            auth_info_2=X_EMAIL,
            password=X_PASSWORD,
            cookies_file=X_COOKIES_PATH
        )
    except Exception as e:
        print(f"Erro no login do X: {e}")
        return []

    try:
        tweets = await client.search_tweet(query, product='Top', count=max_results)
    except TooManyRequests:
        return []

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)
    results = []

    for tweet in tweets:
        tweet_time = datetime.strptime(tweet.created_at, '%a %b %d %H:%M:%S %z %Y')
        
        if tweet_time < cutoff_time:
            continue
            
        if tweet.favorite_count < MIN_LIKES or tweet.retweet_count < MIN_RETWEETS:
            continue

        results.append({
            "url": f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}",
            "text": tweet.text,
            "author": tweet.user.screen_name,
            "created_at": tweet_time.isoformat(),
            "likes": tweet.favorite_count,
            "retweets": tweet.retweet_count
        })

    return results

if __name__ == "__main__":
    import asyncio
    
    async def main():
        noticias = await scrape_x("Bitcoin")
        for n in noticias:
            print(n)
            
    asyncio.run(main())