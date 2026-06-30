import os
import asyncio
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.errors import FloodWaitError

load_dotenv()

class TelegramScraper:
    
    TARGET_CHANNELS = [
        "Cointelegraph",            # Notícias cripto (ativo)
        "CoinDesk",                 # Notícias cripto (ativo)
        "BitcoinMagazine",          # Notícias Bitcoin (ativo)
        "binanceannouncements",     # Anúncios Binance (ativo)
        "CryptoBoomNews",           # Notícias cripto (ativo)
        "CryptoCurrency_News",      # Notícias cripto (ativo)
        "CryptoTownEU",             # Notícias cripto (ativo)
        # "whale_alert",            # ⚠️ Parou de postar em Mar/2020 - desativado
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

        for channel in chans:
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
        """
        Coleta mensagens históricas dos canais do Telegram.
        Inclui delays entre canais e batches para evitar ban.
        
        Args:
            start_year, start_month: Início do período
            end_year, end_month: Fim do período  
            channels: Lista de canais (usa TARGET_CHANNELS se None)
        """
        client = TelegramClient('crypto_scraper_session', int(self.tg_api_id), self.tg_api_hash)

        try:
            print("Conectando ao Telegram para coleta histórica...")
            await client.start(phone=self.tg_phone)
        except Exception as e:
            print(f"Erro ao conectar ao Telegram: {e}")
            return []

        start_date = datetime(start_year, start_month, 1, tzinfo=timezone.utc)

        # Calcula o último momento do mês final para incluir todas as mensagens do período
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(end_year, end_month + 1, 1, tzinfo=timezone.utc)

        all_results = []
        chans = channels or self.TARGET_CHANNELS

        for i, channel in enumerate(chans):
            try:
                print(f"\n📡 [{i+1}/{len(chans)}] Coletando histórico do canal: @{channel}")
                channel_count = 0
                
                async for message in client.iter_messages(channel, offset_date=end_date, limit=None):
                    # Tenta .text primeiro, depois .message (captura msgs formatadas/entidades)
                    text = message.text or message.message or ""
                    
                    msg_time = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None else message.date

                    # Se a primeira mensagem já está fora do período, canal está inativo nessa época
                    if channel_count == 0 and msg_time < start_date:
                        print(f"  ⚠️  Canal @{channel} sem mensagens no período solicitado (última msg: {msg_time.strftime('%Y-%m-%d')})")
                        break
                    
                    if not text or len(text.strip()) < 5:
                        continue

                    if msg_time < start_date:
                        break

                    all_results.append({
                        "url": f"https://t.me/{channel}/{message.id}",
                        "text": text.strip(),
                        "author": channel,
                        "created_at": msg_time.isoformat(),
                        "views": getattr(message, 'views', 0) or 0,
                        "forwards": getattr(message, 'forwards', 0) or 0
                    })
                    channel_count += 1
                    
                    # Delay a cada 500 mensagens para não sobrecarregar
                    if channel_count % 500 == 0:
                        print(f"  📥 {channel_count} mensagens coletadas de @{channel}...")
                        await asyncio.sleep(1)
                
                print(f"  ✅ @{channel}: {channel_count} mensagens coletadas")
                    
            except FloodWaitError as e:
                print(f"  ⚠️  FloodWaitError no canal @{channel}: aguardando {e.seconds}s (exigido pelo Telegram)")
                await asyncio.sleep(e.seconds)
            except Exception as e:
                print(f"  ❌ Erro ao coletar histórico do canal @{channel}: {e}")
            
            # Delay entre canais para evitar ban (2s)
            if i < len(chans) - 1:
                await asyncio.sleep(2)

        await client.disconnect()
        print(f"\n📊 Total geral: {len(all_results)} mensagens coletadas de {len(chans)} canais")
        return all_results

