import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.dataset import CryptoMultimodalDataset
from src.models.integrated_model import IntegratedCryptoModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path_binance = "../data/processed/binance_btc_1h_features.csv"
    data_path_sentiment = "../data/processed/crypto_sentiment_1h.csv"
    weights_dir = "../weights"
    os.makedirs(weights_dir, exist_ok=True)

    dataset = CryptoMultimodalDataset(
        binance_csv=data_path_binance,
        sentiment_csv=data_path_sentiment,
        seq_length=60,
        is_training=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = IntegratedCryptoModel(
        lstm_input_size=8,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_rules=10,
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for seq_batch, sent_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(device)
            sent_batch = sent_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            predictions = model(seq_batch, sent_batch)
            
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seq_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for seq_batch, sent_batch, target_batch in val_loader:
                seq_batch = seq_batch.to(device)
                sent_batch = sent_batch.to(device)
                target_batch = target_batch.to(device)
                
                predictions = model(seq_batch, sent_batch)
                loss = criterion(predictions, target_batch)
                val_loss += loss.item() * seq_batch.size(0)
                
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_crypto_model.pth"))

if __name__ == "__main__":
    train_model()