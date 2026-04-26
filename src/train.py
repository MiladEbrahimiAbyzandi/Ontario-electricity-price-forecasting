import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import joblib
import copy
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from src.config import MODELS_DIR, DATA_CFG, TRAIN_CFG, TrainConfig
from src.model import GRURegressor

from src.preprocess import prepare_data


#---Reproducibility--------------------------
def reset_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

#---Dataset---------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def make_loaders(X_train, y_train,
                 X_val,   y_val,
                 X_test,  y_test,
                 batch_size: int = 32):
   
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

#---training-----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
    return running_loss/ len(loader.dataset)

def train_model(model, train_loader, val_loader,
                device: str='cpu', epochs: int=100,
                learning_rate: float = 1e-2,
                patience: int=10):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss":[], "val_loss":[]}
    for epoch in range(epochs):
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device )
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        mlflow.log_metrics(
            {"train_loss": train_loss , "val_loss": val_loss},
            step=epoch,
        )

        print(f"epoch: {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict()) 
            patience_counter = 0
        else: 
            patience_counter +=1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    return model, history, best_val_loss


def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy().flatten())
    return np.array(preds)

def run_training(df: pd.DataFrame, cfg: TrainConfig = TRAIN_CFG, seq_length: int = DATA_CFG.seq_length):
    """
    Full training pipeline - takes a raw master dataframe,
    preprocesses it, trains GRU for each horizon, logs to MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with columns: date, hour, hoep,
        market_demand, ontario_demand, nuclear, gas,
        hydro, wind, solar, biofuel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # preprocessing handles splitting, scaling, sequence creation
    print("Preprocessing data...")
    data, scaler_y = prepare_data(df, seq_length=seq_length)

    mlflow.set_experiment("hoep-gru-forecasting")

    for h in DATA_CFG.horizons:
        X_train, y_train, X_val, y_val, X_test, y_test = data[h]
        train_loader, val_loader, test_loader = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=cfg.batch_size,
        )

        # each horizon is one MLflow run
        with mlflow.start_run(run_name=f"GRU_h{h}"):

            # --log hyperparameters---------------
            mlflow.log_params({
                "horizon": h,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "patience": cfg.patience,
                "seq_length": seq_length,
            })

            # -- train -----------------
            model = GRURegressor(
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ).to(device)
            model, history, best_val_loss = train_model(
                model, train_loader, val_loader,
                device=device,
                epochs=cfg.epochs,
                learning_rate=cfg.learning_rate,
                patience=cfg.patience,
            )

            #--- evaluate on test set --------------
            test_preds_scaled = predict_model(model, test_loader, device)
            test_preds = scaler_y.inverse_transform(
                test_preds_scaled.reshape(-1, 1)
            ).flatten()
            y_test_orig = scaler_y.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            mae = float(np.mean(np.abs(y_test_orig - test_preds)))
            rmse = float(np.sqrt(np.mean((y_test_orig - test_preds) ** 2)))
            mape = float(
                np.mean(np.abs((y_test_orig - test_preds) / (y_test_orig + 1e-8))) * 100
            )

            # --- log test metrics----------------
            mlflow.log_metrics({
                "test_mae": mae,
                "test_rmse": rmse,
                "test_mape": mape,
                "best_val_loss": best_val_loss,
            })

            print(f"Horizon {h} -- MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%")

            # log model artifact 
            mlflow.log_artifact(
                local_path=str(MODELS_DIR / f"gru_h{h}.pt"),
                artifact_path="models",
            )
    
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            print(f"Open MLflow UI, compare metrics, register good versions manually")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GRU forecasting models")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to master dataset parquet file")
    parser.add_argument("--seq-length",    type=int,   default=DATA_CFG.seq_length)
    parser.add_argument("--hidden-size",   type=int,   default=TRAIN_CFG.hidden_size)
    parser.add_argument("--num-layers",    type=int,   default=TRAIN_CFG.num_layers)
    parser.add_argument("--dropout",       type=float, default=TRAIN_CFG.dropout)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_CFG.learning_rate)
    parser.add_argument("--batch-size",    type=int,   default=TRAIN_CFG.batch_size)
    parser.add_argument("--epochs",        type=int,   default=TRAIN_CFG.epochs)
    parser.add_argument("--patience",      type=int,   default=TRAIN_CFG.patience)
    args = parser.parse_args()

    cfg = TrainConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    df = pd.read_parquet(args.data_path)
    run_training(df, cfg, seq_length=args.seq_length)
