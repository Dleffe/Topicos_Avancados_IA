import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from collections import defaultdict
from datetime import timezone, timedelta
from dateutil import parser as dateutil_parser
import pandas as pd
import asyncio
import argparse
import torch
from transformers import pipeline
from tqdm import tqdm
tqdm.monitor_interval = 0

from src.data_pipeline.gnews_scraper import GoogleNewsScraper
from src.data_pipeline.reddit_scraper import RedditScraper
from src.data_pipeline.telegram_scraper import TelegramScraper

PLATFORM_WEIGHTS = {"Telegram": 1.2, "Google News": 1.0, "Reddit": 0.8}
MODEL_NAME = "ElKulako/cryptobert"


def _parse_dt(raw):
    if not raw:
        return None
    try:
        dt = dateutil_parser.parse(str(raw))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_json(filepath: str) -> list:
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read {filepath}.")
        return []


def save_json(data: list, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


async def scrape_sources(mode: str, start_date: str, end_date: str, query: str,
                         subreddits: list, raw_dir: str,
                         disable_telegram: bool, disable_reddit: bool, disable_gnews: bool):
    os.makedirs(raw_dir, exist_ok=True)
    telegram_path = os.path.join(raw_dir, "telegram_raw.json")
    reddit_path   = os.path.join(raw_dir, "reddit_raw.json")
    gnews_path    = os.path.join(raw_dir, "gnews_raw.json")

    print(f"\n[FASE 1] Extraindo dados — modo: {mode}")

    if disable_telegram:
        print("1/3 -> Telegram desabilitado.")
        save_json([], telegram_path)
    else:
        print("1/3 -> Procurando Telegram...")
        scraper = TelegramScraper()
        if mode == "historical":
            s = pd.to_datetime(start_date)
            e = pd.to_datetime(end_date)
            data = await scraper.get_historical_data(s.year, s.month, e.year, e.month)
        else:
            data = await scraper.get_current_data(query=query)
        save_json(data, telegram_path)

    if disable_reddit:
        print("2/3 -> Reddit desabilitado.")
        save_json([], reddit_path)
    else:
        print("2/3 -> Procurando Reddit...")
        scraper = RedditScraper()
        if mode == "historical":
            data = scraper.get_historical_data(start_date, end_date, subreddits)
        else:
            data = scraper.get_current_data(query=query, subreddits=subreddits)
        save_json(data, reddit_path)

    if disable_gnews:
        print("3/3 -> Google News desabilitado.")
        save_json([], gnews_path)
    else:
        print("3/3 -> Procurando Google News...")
        scraper = GoogleNewsScraper()
        if mode == "historical":
            data = scraper.get_historical_data(query, start_date, end_date)
        else:
            data = scraper.get_current_data(query=query)
        save_json(data, gnews_path)

    print("[FASE 1] Concluída.")
    return telegram_path, reddit_path, gnews_path


def build_records(telegram_path: str, reddit_path: str, gnews_path: str) -> list:
    records = []

    for item in load_json(telegram_path):
        dt   = _parse_dt(item.get('created_at'))
        text = item.get('text', '').strip()
        if dt and text:
            records.append({'text': text[:1500], 'created_at': dt, 'platform': 'Telegram'})

    for item in load_json(reddit_path):
        dt   = _parse_dt(item.get('created_at'))
        text = item.get('text', '').strip()
        if dt and text:
            records.append({'text': text[:1500], 'created_at': dt, 'platform': 'Reddit'})

    for item in load_json(gnews_path):
        dt   = _parse_dt(item.get('date'))
        text = f"{item.get('title', '')}. {item.get('desc', '')}".strip()
        if dt and text:
            records.append({'text': text[:1500], 'created_at': dt, 'platform': 'Google News'})

    records.sort(key=lambda r: r['created_at'])

    if not records:
        print("Aviso: nenhum dado extraído das fontes.")

    return records


def run_sentiment(records: list, batch_size: int) -> list:
    if not records:
        return records

    print(f"\n[FASE 2] {len(records)} textos. Carregando CryptoBERT...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    nlp = pipeline("text-classification", model=MODEL_NAME,
                   truncation=True, max_length=512, device=device)

    texts   = [r['text'] for r in records]
    results = list(tqdm(nlp(texts, batch_size=batch_size),
                        total=len(texts), desc="Sentiment"))

    for record, result in zip(records, results):
        direction = 1.0 if result['label'] == 'Bullish' else (-1.0 if result['label'] == 'Bearish' else 0.0)
        weight    = PLATFORM_WEIGHTS.get(record['platform'], 1.0)
        record['sentiment_scalar'] = direction * result['score'] * weight

    return records


def aggregate_to_csv(records: list, output_csv: str) -> None:
    if not records or 'sentiment_scalar' not in records[0]:
        print("Nenhum dado de sentimento. Pulando CSV.")
        return

    print("\n[FASE 3] Agregando em janelas de 1h...")

    hourly_scores  = defaultdict(list)
    hourly_volumes = defaultdict(int)

    for r in records:
        dt  = r['created_at']
        key = dt.replace(minute=0, second=0, microsecond=0)
        hourly_scores[key].append(r['sentiment_scalar'])
        hourly_volumes[key] += 1

    all_hours   = sorted(hourly_scores)
    start_hour  = all_hours[0]
    end_hour    = all_hours[-1]

    rows         = []
    prev_sent    = 0.0
    current_hour = start_hour

    while current_hour <= end_hour:
        scores = hourly_scores.get(current_hour)
        if scores:
            sentiment    = sum(scores) / len(scores)
            volume       = hourly_volumes[current_hour]
            prev_sent    = sentiment
        else:
            sentiment = prev_sent
            volume    = 0

        rows.append({
            'timestamp':        current_hour.strftime('%Y-%m-%d %H:00:00'),
            'sentiment_scalar': sentiment,
            'news_volume':      volume,
        })
        current_hour += timedelta(hours=1)

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df = pd.DataFrame(rows).set_index('timestamp')
    df.to_csv(output_csv)
    print(f"CSV gerado: {output_csv}")
    print(df.head())


async def main():
    parser = argparse.ArgumentParser(description="Pipeline ETL para sentimento de criptomoedas.")
    parser.add_argument("mode", choices=["historical", "current"])
    parser.add_argument("--query",      type=str, default="Bitcoin")
    parser.add_argument("--start_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date",   type=str, default="2023-02-01")
    parser.add_argument("--subreddits", nargs='+',
                        default=["Bitcoin", "CryptoCurrency", "CryptoMarkets"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_csv", type=str, default="data/processed/crypto_sentiment_1h.csv")
    parser.add_argument("--raw_dir",    type=str, default="data/raw_sentiment",
                        help="Diretório para salvar JSONs brutos dos scrapers.")
    parser.add_argument("--disable-telegram", action="store_true")
    parser.add_argument("--disable-reddit",   action="store_true")
    parser.add_argument("--disable-gnews",    action="store_true")
    args = parser.parse_args()

    tg_path, rd_path, gn_path = await scrape_sources(
        args.mode, args.start_date, args.end_date, args.query, args.subreddits,
        args.raw_dir, args.disable_telegram, args.disable_reddit, args.disable_gnews,
    )

    records = build_records(tg_path, rd_path, gn_path)
    records = run_sentiment(records, args.batch_size)
    aggregate_to_csv(records, args.output_csv)


if __name__ == "__main__":
    asyncio.run(main())
