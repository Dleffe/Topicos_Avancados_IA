import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.models.dataset import CryptoMultimodalDataset
from src.models.integrated_model import IntegratedCryptoModel


def validate(weights_path: str = "weights/best_crypto_model_4h_loss.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CryptoMultimodalDataset(
        binance_csv="data/processed/binance_btc_4h_features.csv",
        sentiment_csv="data/processed/crypto_sentiment_4h.csv",
        seq_length=42,
        is_training=False,
        scalers_dir="data/scalers_4h",
    )

    train_size  = int(0.8 * len(dataset))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)

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

    target_mean, target_std = joblib.load("data/scalers_4h/target_scaler.pkl")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for seq_batch, sent_batch, target_batch in val_loader:
            preds = model(seq_batch.to(device), sent_batch.to(device))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_batch.numpy())

    preds_norm   = np.concatenate(all_preds).flatten()
    targets_norm = np.concatenate(all_targets).flatten()

    preds   = preds_norm   * target_std + target_mean
    targets = targets_norm * target_std + target_mean

    _print_metrics("Model (4h)", preds, targets)

    baseline_zero = np.zeros_like(targets)
    baseline_mean = np.full_like(targets, target_mean)
    _print_metrics("Baseline (predict 0)", baseline_zero, targets)
    _print_metrics("Baseline (predict mean)", baseline_mean, targets)


def _print_metrics(name: str, preds: np.ndarray, targets: np.ndarray):
    mae      = np.abs(preds - targets).mean()
    rmse     = np.sqrt(((preds - targets) ** 2).mean())
    corr     = np.corrcoef(preds, targets)[0, 1] if preds.std() > 1e-9 else 0.0
    dir_acc  = (np.sign(preds) == np.sign(targets)).mean()
    pred_std = preds.std()
    tgt_std  = targets.std()

    # Proportion of predictions where |pred| > 0.001 (model is "taking a position")
    active   = (np.abs(preds) > 1e-3).mean()

    print(f"\n{'─' * 48}")
    print(f"  {name}")
    print(f"{'─' * 48}")
    print(f"  Samples:           {len(targets)}")
    print(f"  MAE:               {mae:.6f}")
    print(f"  RMSE:              {rmse:.6f}")
    print(f"  Pearson r:         {corr:+.4f}")
    print(f"  Directional Acc:   {dir_acc:.2%}")
    print(f"  Active preds:      {active:.2%}  (|pred| > 0.001)")
    print(f"  pred_std:          {pred_std:.6f}")
    print(f"  tgt_std:           {tgt_std:.6f}")
    print(f"  std ratio:         {pred_std / (tgt_std + 1e-9):.4f}")


if __name__ == "__main__":
    validate()
