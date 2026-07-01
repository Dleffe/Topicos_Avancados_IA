import os
import argparse
import numpy as np
import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.index.name = 'timestamp'
    df = df[~df.index.isna()].sort_index()
    return df


def _bucket(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Floor a DatetimeIndex to 4h boundaries without using resample."""
    us_per_4h = 4 * 3600 * 1_000_000
    ts_us     = index.astype(np.int64)
    return pd.DatetimeIndex((ts_us // us_per_4h) * us_per_4h, dtype='datetime64[us]')


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )


def _bb_width(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.Series:
    sma   = close.rolling(length).mean()
    sigma = close.rolling(length).std()
    return (2.0 * std * sigma) / sma * 100.0


def resample_ohlcv(input_csv: str, output_csv: str) -> None:
    print(f"Reading {input_csv}...")
    df = _read_csv(input_csv)
    print(f"  {len(df)} rows, index dtype: {df.index.dtype}")

    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df['_b'] = _bucket(df.index)

    print("  Grouping to 4h...")
    g     = df.groupby('_b')
    df_4h = pd.DataFrame({
        'open':   g['open'].first(),
        'high':   g['high'].max(),
        'low':    g['low'].min(),
        'close':  g['close'].last(),
        'volume': g['volume'].sum(),
    })
    df_4h.index.name = 'timestamp'
    df_4h = df_4h.dropna(subset=['close'])
    print(f"  {len(df_4h)} 4h candles before indicators")

    print("  Computing indicators...")
    df_4h['rsi']           = _rsi(df_4h['close'])
    df_4h['macd']          = _macd(df_4h['close'])
    df_4h['bb_width']      = _bb_width(df_4h['close'])
    df_4h['target_return'] = np.log(df_4h['close'].shift(-1) / df_4h['close'])
    df_4h.dropna(inplace=True)

    df_4h.index      = df_4h.index.strftime('%Y-%m-%d %H:%M:%S')
    df_4h.index.name = 'timestamp'

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_4h.to_csv(output_csv)
    print(f"OHLCV 4h saved: {output_csv}  ({len(df_4h)} candles)")


def resample_sentiment(input_csv: str, output_csv: str) -> None:
    print(f"Reading {input_csv}...")
    df = _read_csv(input_csv)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    if 'sentiment_scalar' in df.columns and 'sentiment_mean' not in df.columns:
        df = df.rename(columns={'sentiment_scalar': 'sentiment_mean'})
        df['bullish_ratio']        = 0.0
        df['bearish_ratio']        = 0.0
        df['sentiment_dispersion'] = 0.0

    if 'sentiment_mean' not in df.columns:
        raise ValueError(f"Unrecognised sentiment columns: {list(df.columns)}")

    df = df[['sentiment_mean', 'news_volume', 'bullish_ratio',
             'bearish_ratio', 'sentiment_dispersion']].astype(float).copy()
    df['_b'] = _bucket(df.index)

    print("  Grouping sentiment to 4h...")
    g      = df.groupby('_b')
    vol_4h = g['news_volume'].sum()

    result = {'news_volume': vol_4h}
    for col in ['sentiment_mean', 'bullish_ratio', 'bearish_ratio']:
        df['_w'] = df[col] * df['news_volume']
        weighted        = df.groupby('_b')['_w'].sum()
        simple          = g[col].mean()
        result[col]     = weighted.where(vol_4h > 0, simple) / vol_4h.where(vol_4h > 0, 1.0)

    result['sentiment_dispersion'] = g['sentiment_dispersion'].mean()

    df_4h = pd.DataFrame(result)
    df_4h.index.name = 'timestamp'

    for col in ['sentiment_mean', 'bullish_ratio', 'bearish_ratio']:
        df_4h[col] = df_4h[col].ffill().fillna(0.0)
    df_4h['sentiment_dispersion'] = df_4h['sentiment_dispersion'].fillna(0.0)
    df_4h['news_volume']          = df_4h['news_volume'].fillna(0.0)

    df_4h.index      = df_4h.index.strftime('%Y-%m-%d %H:%M:%S')
    df_4h.index.name = 'timestamp'

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_4h.to_csv(output_csv)
    print(f"Sentiment 4h saved: {output_csv}  ({len(df_4h)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resample 1h CSVs to 4h.")
    parser.add_argument('--binance_in',    default='data/processed/binance_btc_1h_features.csv')
    parser.add_argument('--sentiment_in',  default='data/processed/crypto_sentiment_1h.csv')
    parser.add_argument('--binance_out',   default='data/processed/binance_btc_4h_features.csv')
    parser.add_argument('--sentiment_out', default='data/processed/crypto_sentiment_4h.csv')
    args = parser.parse_args()

    resample_ohlcv(args.binance_in, args.binance_out)
    resample_sentiment(args.sentiment_in, args.sentiment_out)
