import asyncio
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH para garantir que os imports funcionem
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.telegram_scraper import TelegramScraper
from src.data_pipeline.reddit_scraper import RedditScraper

async def test_telegram():
    print("\n--- Testando TelegramScraper (get_historical_data) ---")
    try:
        scraper = TelegramScraper()
        
        # Testando apenas 1 canal para ser rápido, buscando 1 mês (Maio de 2024)
        channels = ["CoinDesk"]
        print(f"Buscando histórico para os canais: {channels} (Maio/2024)")
        
        results = await scraper.get_historical_data(
            start_year=2024, start_month=5, 
            end_year=2024, end_month=5, 
            channels=channels
        )
        
        print(f"Total de mensagens recuperadas: {len(results)}")
        if results:
            print("Amostra do primeiro resultado:")
            print(results[0])
            
            print("\nAmostra do último resultado:")
            print(results[-1])
    except Exception as e:
        print(f"Erro durante o teste do Telegram: {e}")

def test_reddit():
    print("\n--- Testando RedditScraper (get_historical_data) ---")
    try:
        scraper = RedditScraper()
        
        # Testando apenas 1 subreddit para ser rápido, intervalo pequeno de 5 dias
        subreddits = ["Bitcoin"]
        start_date = "2024-05-01"
        end_date = "2024-05-05"
        print(f"Buscando histórico para os subreddits: {subreddits} de {start_date} a {end_date}")
        
        results = scraper.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            subreddits=subreddits
        )
        
        print(f"Total de posts recuperados: {len(results)}")
        if results:
            print("Amostra do primeiro resultado:")
            print(results[0])
            
            print("\nAmostra do último resultado:")
            print(results[-1])
    except Exception as e:
        print(f"Erro durante o teste do Reddit: {e}")

async def main():
    print("Iniciando testes dos scrapers...")
    
    # Executa o teste do Reddit (síncrono)
    test_reddit()
    
    # Executa o teste do Telegram (assíncrono)
    await test_telegram()
    
    print("\nTestes finalizados!")

if __name__ == "__main__":
    asyncio.run(main())
