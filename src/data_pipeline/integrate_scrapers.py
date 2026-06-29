import os
import json
import pandas as pd
import asyncio
import argparse
import tempfile
import torch
from transformers import pipeline
from tqdm import tqdm

from src.data_pipeline.gnews_scraper import GoogleNewsScraper
from src.data_pipeline.reddit_scraper import RedditScraper
from src.data_pipeline.telegram_scraper import TelegramScraper

START_DATE = "2022-01-01"
END_DATE = "2023-06-30"
PLATFORM_WEIGHTS = {"Telegram": 1.2, "Google News": 1.0, "Reddit": 0.8}
MODEL_NAME = "ElKulako/cryptobert"
 
def load_json(filepath: str) -> list:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not decode JSON from {filepath}. File might be empty or corrupt.")
            return []
    return []

def save_json(data: list, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
async def step_1_extract(mode: str, start_date: str, end_date: str, query: str, subreddits: list, temp_dir: str,
                         disable_telegram: bool, disable_reddit: bool, disable_gnews: bool):
    print(f"\n[ETL - FASE 1] Extraindo dados em modo: {mode}")

    telegram_path = os.path.join(temp_dir, "telegram_data.json")
    reddit_path = os.path.join(temp_dir, "reddit_data.json")
    gnews_path = os.path.join(temp_dir, "gnews_data.json")

    if disable_telegram:
        print("1/3 -> Coleta do Telegram desabilitada. Pulando.")
        save_json([], telegram_path)
    else:
        print("1/3 -> Procurando Telegram...")
        telegram_scraper = TelegramScraper()
        if mode == "historical":
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            telegram_data = await telegram_scraper.get_historical_data(
                start_dt.year, start_dt.month, end_dt.year, end_dt.month, channels=None
            )
        else:
            telegram_data = await telegram_scraper.get_current_data(query=query)
        save_json(telegram_data, telegram_path)

    if disable_reddit:
        print("2/3 -> Coleta do Reddit desabilitada. Pulando.")
        save_json([], reddit_path)
    else:
        reddit_scraper = RedditScraper()
        if mode == "historical":
            print("2/3 -> Procurando Reddit (PullPush)...")
            reddit_data = reddit_scraper.get_historical_data(start_date, end_date, subreddits)
        else:
            print("2/3 -> Procurando Reddit...")
            reddit_data = reddit_scraper.get_current_data(query=query, subreddits=subreddits)
        save_json(reddit_data, reddit_path)

    if disable_gnews:
        print("3/3 -> Coleta do Google News desabilitada. Pulando.")
        save_json([], gnews_path)
    else:
        gnews_scraper = GoogleNewsScraper()
        if mode == "historical":
            print("3/3 -> Procurando Google News...")
            gnews_data = gnews_scraper.get_historical_data(query, start_date, end_date)
        else:
            print("3/3 -> Procurando Google News...")
            gnews_data = gnews_scraper.get_current_data(query=query)
        save_json(gnews_data, gnews_path)

    print("[ETL - FASE 1] Concluída. Arquivos JSON temporários gerados.")
    return telegram_path, reddit_path, gnews_path

def unify_data(tg_path: str, rd_path: str, gn_path: str) -> pd.DataFrame:
    all_data = []
    
    tg_data = load_json(tg_path)
    for item in tg_data: item['platform'] = 'Telegram'
    all_data.extend(tg_data)
    
    rd_data = load_json(rd_path)
    for item in rd_data: item['platform'] = 'Reddit'
    all_data.extend(rd_data)
    
    gn_data = load_json(gn_path)
    for item in gn_data: 
        item['text'] = f"{item.get('title', '')}. {item.get('desc', '')}"
        item['created_at'] = item.get('date')
        item['platform'] = 'Google News'
    all_data.extend(gn_data)
    
    if not all_data:
        print("Aviso: Nenhum dado foi extraído das fontes.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['text', 'created_at'])
    df = df.sort_values('created_at')
    df['text'] = df['text'].astype(str)
    
    return df

def calculate_sentiment(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    if df.empty:
        return df

    print(f"Total de {len(df)} textos unificados. Carregando modelo NLP...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    sentiment_pipeline = pipeline(
        "text-classification", 
        model=MODEL_NAME, 
        truncation=True, 
        max_length=512, 
        device=device
    )
    
    print(f"Calculando escores de sentimento em batch (tamanho={batch_size})...")
    
    texts_to_process = df['text'].str.slice(0, 1500).tolist()    
    results = []
    for out in tqdm(sentiment_pipeline(texts_to_process, batch_size=batch_size), total=len(texts_to_process), desc="Inferring Sentiment"):
        results.append(out)
    
    sentiment_scores = []
    for i, result in enumerate(results):
        label = result["label"]
        confidence = result["score"]
        platform = df.iloc[i]['platform']
        weight = PLATFORM_WEIGHTS.get(platform, 1.0)
        direction = 1.0 if label == "Bullish" else (-1.0 if label == "Bearish" else 0.0)
        
        final_score = direction * confidence * weight
        sentiment_scores.append(final_score)
    
    df['sentiment_scalar'] = sentiment_scores
    
    return df

def aggregate_and_save(df: pd.DataFrame, output_csv: str):
    if df.empty or 'sentiment_scalar' not in df.columns:
        print("Nenhum dado de sentimento para agregar. Pulando a gravação do CSV.")
        return

    print("\n[ETL - FASE 3] Agregando dados em janelas de 1 Hora...")
    
    df.set_index('created_at', inplace=True)
    
    hourly_df = df.resample('1h').agg({
        'sentiment_scalar': 'mean',
        'text': 'count'
    }).rename(columns={'text': 'news_volume'})
    
    hourly_df['sentiment_scalar'] = hourly_df['sentiment_scalar'].fillna(method='ffill').fillna(0.0)
    hourly_df['news_volume'] = hourly_df['news_volume'].fillna(0).astype(int)
    
    hourly_df.index = hourly_df.index.strftime('%Y-%m-%d %H:00:00')
    hourly_df.index.name = 'timestamp'

    hourly_df.to_csv(output_csv)
    print(f"\nDataset final gerado: {output_csv}")
    print(hourly_df.head(5))

async def main():
    parser = argparse.ArgumentParser(description="Pipeline de ETL para dados de sentimento de criptomoedas.")
    parser.add_argument(
        "mode", 
        choices=["historical", "current"], 
        help="Modo de execução: 'historical' para um período ou 'current' para dados recentes."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="Bitcoin", 
        help="Termo de busca para as notícias."
    )
    parser.add_argument(
        "--start_date", 
        type=str, 
        default="2023-01-01", 
        help="Data de início (YYYY-MM-DD) para o modo 'historical'."
    )
    parser.add_argument(
        "--end_date", 
        type=str, 
        default="2023-02-01", 
        help="Data de fim (YYYY-MM-DD) para o modo 'historical'."
    )
    parser.add_argument(
        "--subreddits", 
        nargs='+', 
        default=["Bitcoin", "CryptoCurrency", "CryptoMarkets"], 
        help="Lista de subreddits para buscar."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Tamanho do batch para inferência do modelo NLP."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="crypto_sentiment_1h.csv", 
        help="Caminho para o arquivo CSV de saída."
    )
    parser.add_argument(
        "--disable-telegram",
        action="store_true",
        help="Se definido, desabilita a coleta de dados do Telegram."
    )
    parser.add_argument(
        "--disable-reddit",
        action="store_true",
        help="Se definido, desabilita a coleta de dados do Reddit."
    )
    parser.add_argument(
        "--disable-gnews",
        action="store_true",
        help="Se definido, desabilita a coleta de dados do Google News."
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        tg_path, rd_path, gn_path = await step_1_extract(
            args.mode, args.start_date, args.end_date, args.query,
            args.subreddits, temp_dir, args.disable_telegram, args.disable_reddit, args.disable_gnews
        )
        
        print("\n[ETL - FASE 2] Unificando dados e inferindo Sentimento (CryptoBERT)...")
        unified_df = unify_data(tg_path, rd_path, gn_path)
        
        sentiment_df = calculate_sentiment(unified_df, args.batch_size)
        
        aggregate_and_save(sentiment_df, args.output_csv)

if __name__ == "__main__":
    asyncio.run(main())