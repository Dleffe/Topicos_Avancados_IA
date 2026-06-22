import os
import json
import pandas as pd
import asyncio
from transformers import pipeline

from src.data_pipeline.gnews_scraper import GoogleNewsScraper
from src.data_pipeline.reddit_scraper import RedditScraper
from src.data_pipeline.telegram_scraper import TelegramScraper

START_DATE = "2023-01-01"
END_DATE = "2023-02-01"
QUERY = "Bitcoin"
TARGET_SUBREDDITS = ["Bitcoin", "CryptoCurrency", "CryptoMarkets"]

gnews_scraper = GoogleNewsScraper()
reddit_scraper = RedditScraper()
telegram_scraper = TelegramScraper()

def load_json(filepath: str) -> list:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(data: list, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

async def step_1_extract(mode: str = "historical"):
    print(f"\nModo: {mode}")
    
    if mode == "historical":
        print("1/3 -> Procurando Telegram...")
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        telegram_data = await telegram_scraper.get_historical_data(
            start_dt.year, start_dt.month, 
            end_dt.year, end_dt.month
        )
        save_json(telegram_data, "temp_telegram.json")
        
        print("2/3 -> Procurando Reddit (PullPush)...")
        reddit_data = reddit_scraper.get_historical_data(START_DATE, END_DATE, TARGET_SUBREDDITS)
        save_json(reddit_data, "temp_reddit.json")
        
        print("3/3 -> Procurando Google News...")
        gnews_data = gnews_scraper.get_historical_data(QUERY, START_DATE, END_DATE)
        save_json(gnews_data, "temp_gnews.json")
    else:
        print("1/3 -> Procurando Telegram...")
        telegram_data = await telegram_scraper.get_current_data(query=QUERY)
        save_json(telegram_data, "temp_telegram.json")
        
        print("2/3 -> Procurando Reddit...")
        reddit_data = reddit_scraper.get_current_data(query=QUERY, subreddits=TARGET_SUBREDDITS)
        save_json(reddit_data, "temp_reddit.json")
        
        print("3/3 -> Procurando Google News...")
        gnews_data = gnews_scraper.get_current_data(query=QUERY)
        save_json(gnews_data, "temp_gnews.json")
    
    print("[ETL - FASE 1] Concluída. Arquivos JSON temporários gerados.")

def step_2_transform_and_load():
    print("\n[ETL - FASE 2] Unificando dados e inferindo Sentimento (CryptoBERT)...")
    
    all_data = []
    
    tg_data = load_json("temp_telegram.json")
    for item in tg_data: item['platform'] = 'Telegram'
    all_data.extend(tg_data)
    
    rd_data = load_json("temp_reddit.json")
    for item in rd_data: item['platform'] = 'Reddit'
    all_data.extend(rd_data)
    
    gn_data = load_json("temp_gnews.json")
    for item in gn_data: item['platform'] = 'Google News'
    all_data.extend(gn_data)
    
    if not all_data:
        print("Erro: Nenhum dado extraído.")
        return
        
    df = pd.DataFrame(all_data)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df = df.dropna(subset=['text', 'created_at'])
    df = df.sort_values('created_at')
    
    print(f"Total de {len(df)} textos unificados. Carregando modelo NLP...")
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Specify device=0 to ensure GPU usage. Adjust batch_size based on VRAM.
    sentiment_pipeline = pipeline("text-classification", model="ElKulako/cryptobert", truncation=True, max_length=512, device=0)
    
    platform_weights = {"Telegram": 1.2, "Google News": 1.0, "Reddit": 0.8}
    
    print("Calculando escores de sentimento em batch...")
    
    texts_to_process = df['text'].str.slice(0, 1500).tolist()
    results = sentiment_pipeline(texts_to_process, batch_size=32)
    
    sentiment_scores = []
    for i, result in enumerate(results):
        label = result["label"]
        confidence = result["score"]
        platform = df.iloc[i]['platform']
        weight = platform_weights.get(platform, 1.0)
        direction = 1.0 if label == "Bullish" else (-1.0 if label == "Bearish" else 0.0)
        final_score = direction * confidence * weight
        sentiment_scores.append(final_score)
            
    df['sentiment_scalar'] = sentiment_scores
    
    print("\n[ETL - FASE 3] Agregando dados em janelas de 1 Hora...")
    
    df.set_index('created_at', inplace=True)
    
    hourly_df = df.resample('1h').agg({
        'sentiment_scalar': 'mean',
        'text': 'count'
    }).rename(columns={'text': 'news_volume'})
    
    hourly_df['sentiment_scalar'] = hourly_df['sentiment_scalar'].fillna(0.0)
    hourly_df['news_volume'] = hourly_df['news_volume'].fillna(0)
    
    hourly_df.index = hourly_df.index.strftime('%Y-%m-%d %H:00:00')
    hourly_df.index.name = 'timestamp'
    
    output_csv = "crypto_sentiment_1h.csv"
    hourly_df.to_csv(output_csv)
    print(f"\n✅ SUCESSO! Dataset final gerado: {output_csv}")
    print(hourly_df.head(5))

if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "historical"
    
    if mode not in ["historical", "current"]:
        print("Uso: python -m src.data_pipeline.integrate_scrapers [historical|current]")
        sys.exit(1)
    
    asyncio.run(step_1_extract(mode=mode))
    step_2_transform_and_load()