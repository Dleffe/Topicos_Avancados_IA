import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class CryptoMultimodalDataset(Dataset):
    def __init__(self, binance_csv: str, sentiment_csv: str, seq_length: int = 60, is_training: bool = True):
        df_kline = pd.read_csv(binance_csv, index_col='timestamp', parse_dates=True)
        df_sent = pd.read_csv(sentiment_csv, index_col='timestamp', parse_dates=True)
        
        df = df_kline.join(df_sent, how='inner')
        df.dropna(inplace=True)
        
        self.targets = df['target_return'].values
        self.sentiments = df['sentiment_scalar'].values
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width']
        self.temporal_features = df[feature_cols].values
        
        if is_training:
            self.scaler = StandardScaler()
            self.temporal_features = self.scaler.fit_transform(self.temporal_features)
            joblib.dump(self.scaler, 'temporal_scaler.pkl') 
        else:
            self.scaler = joblib.load('temporal_scaler.pkl')
            self.temporal_features = self.scaler.transform(self.temporal_features)
            
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.temporal_features) - self.seq_length
        
    def __getitem__(self, idx):
        x_seq = self.temporal_features[idx : idx + self.seq_length]
        
        current_idx = idx + self.seq_length - 1
        x_sent = self.sentiments[current_idx]
        y_target = self.targets[current_idx]
        
        return (
            torch.tensor(x_seq, dtype=torch.float32), 
            torch.tensor([x_sent], dtype=torch.float32), 
            torch.tensor([y_target], dtype=torch.float32)
        )

if __name__ == "__main__":
    dataset = CryptoMultimodalDataset("binance_btc_1h_features.csv", "crypto_sentiment_1h.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    seq, sent, target = next(iter(dataloader))
    
    print("\n📦 Estrutura do Batch (Batch Size = 32):")
    print(f"Série Temporal (Bi-LSTM): {seq.shape} -> [Lote, 60 Horas, 8 Features]")
    print(f"Sentimento (Neuro-Fuzzy): {sent.shape} -> [Lote, 1 Escalar]")
    print(f"Alvo (Retorno Log):       {target.shape} -> [Lote, 1 Escalar]")