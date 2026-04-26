# src/preprocess.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import SCALER_DIR, DATA_CFG


# ── Splitting ──────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into train, validation, and test sets.
    Matches the split used in notebook 03:
        train : up to 2024-12-31
        val   : 2025-01-01 to 2025-03-31
        test  : 2025-04-01 to 2025-04-30
    """
    train = df[df["date"] <= pd.Timestamp("2024-12-31")]
    val   = df[
        (df["date"] >= pd.Timestamp("2025-01-01")) &
        (df["date"] <= pd.Timestamp("2025-03-31"))
    ]
    test  = df[
        (df["date"] >= pd.Timestamp("2025-04-01")) &
        (df["date"] <= pd.Timestamp("2025-04-30"))
    ]
    return train, val, test


# ── Scaling ────────────────────────────────────────────────────────────

def fit_scalers(
    train: pd.DataFrame,
) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit scaler_X on feature columns and scaler_y on target column
    using training data only — never fit on val or test.
    Saves scalers to artifacts/scalers/ for use at inference time.
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(train[DATA_CFG.feature_cols])
    scaler_y.fit(train[DATA_CFG.target_col].values.reshape(-1, 1))

    joblib.dump(scaler_X, SCALER_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, SCALER_DIR / "scaler_y.pkl")

    return scaler_X, scaler_y


def apply_scaling(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> tuple:
    """
    Apply fitted scalers to all splits.
    scaler_X transforms features, scaler_y transforms target.
    """
    X_train = scaler_X.transform(train[DATA_CFG.feature_cols])
    X_val   = scaler_X.transform(val[DATA_CFG.feature_cols])
    X_test  = scaler_X.transform(test[DATA_CFG.feature_cols])

    y_train = scaler_y.transform(train[DATA_CFG.target_col].values.reshape(-1, 1)).flatten()
    y_val   = scaler_y.transform(val[DATA_CFG.target_col].values.reshape(-1, 1)).flatten()
    y_test  = scaler_y.transform(test[DATA_CFG.target_col].values.reshape(-1, 1)).flatten()

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Sequence creation ──────────────────────────────────────────────────

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = DATA_CFG.seq_length,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of seq_length over X and y.
    Each sample is seq_length consecutive rows of X,
    target is the value horizon steps after the window ends.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length - horizon + 1):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length + horizon - 1])
    return np.array(X_seq), np.array(y_seq)


def create_sequences_with_context(
    X_context: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = DATA_CFG.seq_length,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For val and test sets — uses the last seq_length rows of the
    previous split as context so no information is lost at boundaries.
    """
    context    = X_context[-seq_length:]
    X_combined = np.concatenate([context, X], axis=0)

    X_seq, y_seq = [], []
    for i in range(len(X) - horizon + 1):
        X_seq.append(X_combined[i : i + seq_length])
        y_seq.append(y[i + horizon - 1])
    return np.array(X_seq), np.array(y_seq)


# ── Main entry point ───────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    seq_length: int = DATA_CFG.seq_length,
) -> dict[int, tuple[np.ndarray, ...]]:
    """
    Full preprocessing pipeline from raw dataframe to
    model-ready sequences for all three horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with columns: date, hour, hoep,
        market_demand, ontario_demand, nuclear, gas,
        hydro, wind, solar, biofuel

    Returns
    -------
    dict mapping horizon → (X_train, y_train, X_val, y_val, X_test, y_test)
    each X has shape (n_samples, SEQ_LENGTH, n_features)
    each y has shape (n_samples,)
    """
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    # 1. split
    train, val, test = split_data(df)

    # 2. fit scalers on train only, save to disk
    scaler_X, scaler_y = fit_scalers(train)

    # 3. scale all splits
    X_train, y_train, X_val, y_val, X_test, y_test = apply_scaling(
        train, val, test, scaler_X, scaler_y
    )

    # 4. create sequences for each horizon
    data = {}
    for h in DATA_CFG.horizons:
        X_tr, y_tr = create_sequences(X_train, y_train, seq_length=seq_length, horizon=h)
        X_v,  y_v  = create_sequences_with_context(X_train, X_val,  y_val,  seq_length=seq_length, horizon=h)
        X_te, y_te = create_sequences_with_context(X_val,   X_test, y_test, seq_length=seq_length, horizon=h)

        data[h] = (X_tr, y_tr, X_v, y_v, X_te, y_te)

    return data, scaler_y