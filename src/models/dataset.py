import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class CryptoMultimodalDataset(Dataset):
    def __init__(self, binance_csv: str, sentiment_csv: str, seq_length: int = 60,
                 is_training: bool = True, scalers_dir: str = 'data/scalers'):
        df_kline = pd.read_csv(binance_csv, index_col='timestamp', parse_dates=True)
        df_sent  = pd.read_csv(sentiment_csv, index_col='timestamp', parse_dates=True)

        df = df_kline.join(df_sent, how='inner')
        df.dropna(inplace=True)

        sentiment_cols = ['sentiment_mean', 'news_volume', 'bullish_ratio', 'bearish_ratio', 'sentiment_dispersion']
        sentiment_data = df[sentiment_cols].copy()
        sentiment_data['news_volume'] = np.log1p(sentiment_data['news_volume'])
        self.sentiments = sentiment_data.values

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width']
        self.temporal_features = df[feature_cols].values

        os.makedirs(scalers_dir, exist_ok=True)
        _p = lambda name: os.path.join(scalers_dir, name)

        if is_training:
            self.scaler = StandardScaler()
            self.temporal_features = self.scaler.fit_transform(self.temporal_features)
            joblib.dump(self.scaler, _p('temporal_scaler.pkl'))

            self.sentiment_scaler = StandardScaler()
            self.sentiments = self.sentiment_scaler.fit_transform(self.sentiments)
            joblib.dump(self.sentiment_scaler, _p('sentiment_scaler.pkl'))

            targets_raw      = df['target_return'].values
            self.target_mean = targets_raw.mean()
            self.target_std  = targets_raw.std() + 1e-8
            joblib.dump((self.target_mean, self.target_std), _p('target_scaler.pkl'))
            self.targets = (targets_raw - self.target_mean) / self.target_std
        else:
            self.scaler = joblib.load(_p('temporal_scaler.pkl'))
            self.temporal_features = self.scaler.transform(self.temporal_features)

            self.sentiment_scaler = joblib.load(_p('sentiment_scaler.pkl'))
            self.sentiments = self.sentiment_scaler.transform(self.sentiments)

            self.target_mean, self.target_std = joblib.load(_p('target_scaler.pkl'))
            self.targets = (df['target_return'].values - self.target_mean) / self.target_std
            
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.temporal_features) - self.seq_length
        
    def __getitem__(self, idx):
        x_seq = self.temporal_features[idx : idx + self.seq_length]
        
        current_idx = idx + self.seq_length - 1
        x_sent   = self.sentiments[current_idx]   # shape (5,)
        y_target = self.targets[current_idx]

        return (
            torch.tensor(x_seq,      dtype=torch.float32),
            torch.tensor(x_sent,     dtype=torch.float32),
            torch.tensor([y_target], dtype=torch.float32),
        )

if __name__ == "__main__":
    dataset = CryptoMultimodalDataset("binance_btc_1h_features.csv", "crypto_sentiment_1h.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    seq, sent, target = next(iter(dataloader))
    
    print("\n📦 Estrutura do Batch (Batch Size = 32):")
    print(f"Série Temporal (Bi-LSTM): {seq.shape} -> [Lote, 60 Horas, 8 Features]")
    print(f"Sentimento (Neuro-Fuzzy): {sent.shape} -> [Lote, 1 Escalar]")
    print(f"Alvo (Retorno Log):       {target.shape} -> [Lote, 1 Escalar]")