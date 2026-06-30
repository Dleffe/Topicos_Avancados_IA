import pandas as pd
import requests
from datetime import datetime, timezone
import pandas_ta as ta
import time
import numpy as np

def fetch_binance_ohlcv(symbol, interval, start_str, end_str):
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    all_klines = []
    
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": limit
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.5)
        
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    return df[numeric_cols]

def process_technical_indicators(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands.filter(like='BBU').iloc[:, 0]
    df['bb_middle'] = bbands.filter(like='BBM').iloc[:, 0]
    df['bb_lower'] = bbands.filter(like='BBL').iloc[:, 0]
    df['bb_width'] = bbands.filter(like='BBB').iloc[:, 0]
    
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    import os
    out_path = "data/processed/binance_btc_1h_features.csv"
    os.makedirs("data/processed", exist_ok=True)

    print("Baixando dados da Binance...")
    df_raw = fetch_binance_ohlcv("BTCUSDT", "1h", "2022-01-01", "2023-07-01")

    print("Calculando indicadores técnicos...")
    df_processed = process_technical_indicators(df_raw)

    df_processed.index = df_processed.index.strftime('%Y-%m-%d %H:00:00')
    df_processed.to_csv(out_path)
    print(f"Arquivo salvo em {out_path}")