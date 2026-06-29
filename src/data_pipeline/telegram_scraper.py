import os
import asyncio
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.errors import FloodWaitError
from tqdm.asyncio import tqdm

load_dotenv()

class TelegramScraper:
    
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
    
    def __init__(self):
        self.hours_window = int(os.getenv("HOURS_WINDOW", 24))
        self.tg_api_id = os.getenv("TG_API_ID")
        self.tg_api_hash = os.getenv("TG_API_HASH")
        self.tg_phone = os.getenv("TG_PHONE")
        
        if not self.tg_api_id or not self.tg_api_hash:
            raise ValueError("TG_API_ID ou TG_API_HASH não configurados no .env")
    
    async def get_current_data(self, query: str = None, channels: list = None) -> list:
        client = TelegramClient('crypto_scraper_session', int(self.tg_api_id), self.tg_api_hash)
        
        try:
            print("Conectando ao Telegram...")
            await client.start(phone=self.tg_phone)
        except Exception as e:
            print(f"Erro ao conectar ao Telegram: {e}")
            return []

        results = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.hours_window)
        chans = channels or self.TARGET_CHANNELS

        for channel in tqdm(chans, desc="Scanning Telegram Channels (Current)"):
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
    
    async def get_historical_data(self, start_year: int, start_month: int, end_year: int, end_month: int, channels: list = None) -> list:
        client = TelegramClient('crypto_scraper_session', int(self.tg_api_id), self.tg_api_hash)
        await client.start(phone=self.tg_phone)

        start_date = datetime(start_year, start_month, 1, tzinfo=timezone.utc)
        end_date = datetime(end_year, end_month, 1, tzinfo=timezone.utc)

        all_results = []
        chans = channels or self.TARGET_CHANNELS

        for channel in tqdm(chans, desc="Scanning Telegram Channels (Historical)"):
            try:
                async for message in client.iter_messages(channel, offset_date=end_date):
                    if not message.text:
                        continue

                    msg_time = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date

                    if msg_time < start_date:
                        break

                    all_results.append({
                        "url": f"https://t.me/{channel}/{message.id}",
                        "text": message.text,
                        "author": channel,
                        "created_at": msg_time.isoformat(),
                        "views": message.views if hasattr(message, 'views') else 0,
                        "forwards": message.forwards if hasattr(message, 'forwards') else 0
                    })
            except FloodWaitError as e:
                await asyncio.sleep(e.seconds)
            except Exception:
                pass

        await client.disconnect()
        return all_results
