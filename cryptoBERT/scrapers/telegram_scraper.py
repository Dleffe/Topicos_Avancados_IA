import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError

load_dotenv()

TG_API_ID = os.getenv("TG_API_ID")
TG_API_HASH = os.getenv("TG_API_HASH")
TG_PHONE = os.getenv("TG_PHONE")

HOURS_WINDOW = int(os.getenv("HOURS_WINDOW", 24))

# Canais de exemplo para buscar sinais de whales ou notícias macro
TARGET_CHANNELS = [
    "whale_alert",
    "Cointelegraph",
    "binanceannouncements",
    "CoinDesk",
    "BitcoinMagazine",
    "CryptoBoomNews",
    "CryptoCurrency_News",
    "ICO_Drops",
    "CryptoTownEU"
]

async def scrape_telegram(query: str = None) -> list[dict]:

    if not TG_API_ID or not TG_API_HASH:
        print("Erro: TG_API_ID ou TG_API_HASH não configurados no .env")
        return []

    client = TelegramClient('crypto_scraper_session', int(TG_API_ID), TG_API_HASH)
    
    try:
        print("Conectando ao Telegram...")
        await client.start(phone=TG_PHONE)
    except Exception as e:
        print(f"Erro ao conectar ou fazer login: {e}")
        return []

    results = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=HOURS_WINDOW)

    for channel in TARGET_CHANNELS:
        try:
            print(f"Buscando no canal: @{channel}")
            async for message in client.iter_messages(channel, limit=500):
                if not message.text:
                    continue
                
                msg_time = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date
                if msg_time < cutoff_time:
                    break
                    
                if query and query.lower() not in message.text.lower():
                    continue
                    
                results.append({
                    "url": f"https://t.me/{channel}/{message.id}",
                    "text": message.text,
                    "author": channel,
                    "created_at": msg_time.isoformat(),
                    "views": message.views if hasattr(message, 'views') else 0,
                    "forwards": message.forwards if hasattr(message, 'forwards') else 0
                })
        except Exception as e:
            print(f"Erro ao ler o canal {channel}: {e}")

    await client.disconnect()
    return results

if __name__ == "__main__":
    import asyncio
    
    async def main():
        noticias = await scrape_telegram("Bitcoin")
        if noticias:
            print(json.dumps(noticias[:2], indent=4, ensure_ascii=False))
        else:
            print("Nenhuma mensagem encontrada.")
            
    asyncio.run(main())
