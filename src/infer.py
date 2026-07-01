import os
import numpy as np
import torch
import joblib
import pandas as pd
from src.models.integrated_model import IntegratedCryptoModel

def run_inference(binance_csv_path, sentiment_csv_path, scaler_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IntegratedCryptoModel(
        lstm_input_size=8,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_rules=10,
        dropout=0.2,
        sentiment_input_dim=5,
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    df_kline = pd.read_csv(binance_csv_path, index_col='timestamp', parse_dates=True)
    df_sent = pd.read_csv(sentiment_csv_path, index_col='timestamp', parse_dates=True)
    df = df_kline.join(df_sent, how='inner').dropna()

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width']
    recent_data = df[feature_cols].tail(60).values

    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(recent_data)

    target_mean, target_std = joblib.load('data/scalers/target_scaler.pkl')

    sentiment_cols = ['sentiment_mean', 'news_volume', 'bullish_ratio', 'bearish_ratio', 'sentiment_dispersion']
    recent_sentiment = df[sentiment_cols].iloc[-1].values.astype(np.float32)
    recent_sentiment[1] = np.log1p(recent_sentiment[1])
    sentiment_scaler = joblib.load('data/scalers/sentiment_scaler.pkl')
    recent_sentiment = sentiment_scaler.transform(recent_sentiment.reshape(1, -1))[0]

    seq_tensor  = torch.tensor(scaled_data,       dtype=torch.float32).unsqueeze(0).to(device)
    sent_tensor = torch.tensor(recent_sentiment,  dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(seq_tensor, sent_tensor)

    return prediction.item() * target_std + target_mean

if __name__ == "__main__":
    binance_data = "data/processed/binance_btc_1h_features.csv"
    sentiment_data = "data/processed/crypto_sentiment_1h.csv"
    scaler_file = "data/scalers/temporal_scaler.pkl"
    weights_file = "weights/best_crypto_model.pth"
    
    predicted_return = run_inference(binance_data, sentiment_data, scaler_file, weights_file)
    print(predicted_return)