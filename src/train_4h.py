import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.dataset import CryptoMultimodalDataset
from src.models.integrated_model import IntegratedCryptoModel


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path_binance   = "data/processed/binance_btc_4h_features.csv"
    data_path_sentiment = "data/processed/crypto_sentiment_4h.csv"
    scalers_dir         = "data/scalers_4h"
    weights_dir         = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    dataset = CryptoMultimodalDataset(
        binance_csv=data_path_binance,
        sentiment_csv=data_path_sentiment,
        seq_length=42,       # 7 days of 4h candles
        is_training=True,
        scalers_dir=scalers_dir,
    )

    train_size    = int(0.8 * len(dataset))
    train_dataset = Subset(dataset, range(train_size))
    val_dataset   = Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    model = IntegratedCryptoModel(
        lstm_input_size=8,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        num_rules=10,
        dropout=0.2,
        sentiment_input_dim=5,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6
    )

    epochs        = 50
    best_val_loss = float('inf')
    best_dir_acc  = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for seq_batch, sent_batch, target_batch in train_loader:
            seq_batch    = seq_batch.to(device)
            sent_batch   = sent_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            predictions = model(seq_batch, sent_batch)
            loss        = criterion(predictions, target_batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * seq_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss  = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for seq_batch, sent_batch, target_batch in val_loader:
                preds     = model(seq_batch.to(device), sent_batch.to(device))
                val_loss += criterion(preds, target_batch.to(device)).item() * seq_batch.size(0)
                all_preds.append(preds.cpu())
                all_tgts.append(target_batch)
        val_loss /= len(val_loader.dataset)

        preds_cat = torch.cat(all_preds).flatten()
        tgts_cat  = torch.cat(all_tgts).flatten()
        dir_acc   = (preds_cat.sign() == tgts_cat.sign()).float().mean().item()
        pred_std  = preds_cat.std().item()
        tgt_std   = tgts_cat.std().item()

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} "
            f"| dir_acc={dir_acc:.2%} | pred_std={pred_std:.6f} | tgt_std={tgt_std:.6f} | grad_norm={grad_norm:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_crypto_model_4h_loss.pth"))

        if dir_acc > best_dir_acc:
            best_dir_acc = dir_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_crypto_model_4h_dir.pth"))


if __name__ == "__main__":
    train_model()
