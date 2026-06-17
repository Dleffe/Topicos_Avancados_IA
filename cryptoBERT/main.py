import os
import json
import asyncio
import re
from dotenv import load_dotenv

from scrapers.gnews_scraper import scrape_gnews
from scrapers.telegram_scraper import scrape_telegram
from scrapers.reddit_scraper import scrape_reddit

load_dotenv()

async def collect_all_data(query: str):

    print(f"Iniciando coleta de dados para: {query}...")
    
    print("Coletando notícias do Google News...")
    gnews_data = scrape_gnews(query)
    
    print("Coletando mensagens do Telegram...")
    telegram_data = await scrape_telegram(query)
    
    print("Coletando postagens do Reddit...")
    reddit_data = scrape_reddit(query)
    
    unified_results = []
    
    for item in gnews_data:
        raw_content = f"{item.get('title', '')} - {item.get('desc', '')}"
        
        clean_content = re.sub(r'<[^>]+>', ' ', raw_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        unified_results.append({
            "Platform": "Google News",
            "Author": item.get('source', 'News Outlet'),
            "Content": clean_content,
            "TimeStamp": item.get('date', '')
        })
        
    for item in telegram_data:
        unified_results.append({
            "Platform": "Telegram",
            "Author": f"@{item.get('author', 'Unknown')}",
            "Content": item.get('text', ''),
            "TimeStamp": item.get('created_at', '')
        })
        
    for item in reddit_data:
        unified_results.append({
            "Platform": "Reddit",
            "Author": f"u/{item.get('author', 'Unknown')}",
            "Content": item.get('text', ''),
            "TimeStamp": item.get('created_at', '')
        })
        
    output_filename = f"output_{query.replace(' ', '_').lower()}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(unified_results, f, ensure_ascii=False, indent=4)
        
    print(f"\nColeta finalizada! {len(unified_results)} itens consolidados.")
    print(f"Arquivo salvo como: {output_filename}")
    
    from cryptoBERT.sentiment.sentiment import analyze_sentiment
    final_output_filename = f"final_sentiment_{query.replace(' ', '_').lower()}.json"
    analyze_sentiment(output_filename, final_output_filename)
    
if __name__ == "__main__":
    termo_busca = "Bitcoin"
    
    asyncio.run(collect_all_data(termo_busca))
