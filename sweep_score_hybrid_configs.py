from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from DataProcessing import (  # noqa: E402
    Sensors,
    Actuators,
    make_sequences,
    make_sequence_end_timestamps,
    mixed_scale_features,
)
from ModelTraining import LSTMAutoencoder  # noqa: E402


SEQ_LEN = 64
STRIDE = 5
INPUT_DIM = 51
BATCH_SIZE = 512
START_TIME = pd.to_datetime("2015-12-23 12:00:00")

BEST_PARAMS = {
    "score_mode": "max",
    "conv_channels": 64,
    "kernel_size": 3,
    "hidden_size": 64,
    "num_layers": 4,
    "dropout": 0.49521204489910164,
    "bidirectional": True,
    "learning_rate": 0.0002654334209893315,
    "weight_decay": 5.414328125404531e-05,
    "threshold_percentile": 96.87360466212871,
    "top_k": 5,
}

SCORE_CONFIGS = [
    {"score_mode": "max", "top_k": None},
    {"score_mode": "topk", "top_k": 3},
    {"score_mode": "topk", "top_k": 4},
    {"score_mode": "topk", "top_k": 5},
]
THRESHOLD_PERCENTILES = [96.87360466212871, 97.5, 98.0]
SMOOTHING_WINDOWS = [1, 2]
HYBRID_MODES = ["ae_only", "strict_and_high_conf", "loose_or_high_conf"]

RESULT_DIR = ROOT / "results"
DATASET_DIR = ROOT.parent / "Dataset"
MODEL_PATH = RESULT_DIR / "conv_bilstm_autoencoder.pt"
REFERENCE_EVAL_PATH = RESULT_DIR / "evaluation_df_with_rules.parquet"
FULL_TABLE_PATH = RESULT_DIR / "table_score_hybrid_sweep.csv"
PARETO_TABLE_PATH = RESULT_DIR / "table_score_hybrid_pareto.csv"
REFERENCE_TABLE_PATH = RESULT_DIR / "table_score_hybrid_references.csv"
BEST_EVAL_PATH = RESULT_DIR / "evaluation_df_best_score_hybrid.parquet"


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        conv_channels=BEST_PARAMS["conv_channels"],
        kernel_size=BEST_PARAMS["kernel_size"],
        hidden_size=BEST_PARAMS["hidden_size"],
        num_layers=BEST_PARAMS["num_layers"],
        dropout=BEST_PARAMS["dropout"],
        bidirectional=BEST_PARAMS["bidirectional"],
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def prepare_model_inputs(start_time, seq_len: int, stride: int):
    normal_path = DATASET_DIR / "SWaT_Dataset_Normal_v1.parquet"
    attack_path = DATASET_DIR / "SWaT_Dataset_Attack_v1.parquet"

    df_normal = pd.read_parquet(normal_path)
    df_attack = pd.read_parquet(attack_path)
    df_normal["Timestamp"] = pd.to_datetime(df_normal["Timestamp"])
    df_attack["Timestamp"] = pd.to_datetime(df_attack["Timestamp"])

    df_normal = df_normal.loc[df_normal["Timestamp"] >= start_time].copy()

    train_set, other_normal = train_test_split(df_normal, train_size=0.8, shuffle=False)
    val_set, tmp = train_test_split(other_normal, train_size=0.5, shuffle=False)
    test_set = pd.concat([tmp, df_attack], axis=0, ignore_index=True)

    feature_cols = Sensors + Actuators
    train_x, val_x, test_x, _, _ = mixed_scale_features(
        train_set,
        val_set,
        test_set,
        feature_cols,
    )

    val_y = val_set["Label"].to_numpy()
    test_y = test_set["Label"].to_numpy()
    val_timestamps = val_set["Timestamp"].to_numpy()
    test_timestamps = test_set["Timestamp"].to_numpy()

    X_val_seq, _, val_skipped = make_sequences(val_x, val_y, seq_len, stride, val_timestamps)
    X_test_seq, _, test_skipped = make_sequences(test_x, test_y, seq_len, stride, test_timestamps)
    test_window_end, _ = make_sequence_end_timestamps(test_timestamps, seq_len, stride)

    print("\nLightweight preprocessing")
    print(f"train_set = {train_set.shape}")
    print(f"val_set = {val_set.shape}")
    print(f"test_set = {test_set.shape}")
    print(f"X_val_seq = {X_val_seq.shape}, skipped={val_skipped}")
    print(f"X_test_seq = {X_test_seq.shape}, skipped={test_skipped}")

    return X_val_seq, X_test_seq, pd.to_datetime(pd.Series(test_window_end)).reset_index(drop=True)


def get_timestep_mse(model, data_array: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data_array)),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    model.eval()

    with torch.inference_mode():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            reconstruction = model(batch_x)
            timestep_mse = torch.mean((reconstruction - batch_x) ** 2, dim=2)
            rows.append(timestep_mse.cpu().numpy())

    return np.concatenate(rows, axis=0)


def compute_window_score(timestep_mse: np.ndarray, mode: str = "max", top_k: int | None = None) -> np.ndarray:
    if mode == "mean":
        return timestep_mse.mean(axis=1)
    if mode == "max":
        return timestep_mse.max(axis=1)
    if mode == "topk":
        if top_k is None:
            raise ValueError("top_k is required when score_mode='topk'")
        k = max(1, min(int(top_k), timestep_mse.shape[1]))
        topk = np.partition(timestep_mse, -k, axis=1)[:, -k:]
        return topk.mean(axis=1)
    raise ValueError(f"Unsupported score mode: {mode}")


def smooth_positive_runs(pred: np.ndarray, min_run_length: int = 2) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    out = np.zeros_like(pred)
    idx = 0

    while idx < len(pred):
        if pred[idx] == 0:
            idx += 1
            continue

        end = idx
        while end < len(pred) and pred[end] == 1:
            end += 1

        if (end - idx) >= int(min_run_length):
            out[idx:end] = 1
        idx = end

    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Positive Windows": int(y_pred.sum()),
        "Evaluated Windows": int(len(y_true)),
    }


def build_reference_table(reference_df: pd.DataFrame, label_col: str = "label_dynamic") -> pd.DataFrame:
    rows = []
    for setting in [
        "y_pred",
        "hybrid_or_high_conf",
        "hybrid_conservative",
        "hybrid_adaptive_balanced",
        "hybrid_adaptive_f1",
    ]:
        if setting not in reference_df.columns:
            continue
        y_true = reference_df[label_col].to_numpy(dtype=np.int64)
        y_pred = reference_df[setting].to_numpy(dtype=np.int64)
        row = {"Setting": setting}
        row.update(compute_metrics(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["F1-score", "Precision", "Recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_candidate_eval_df(
    base_eval_df: pd.DataFrame,
    errors: np.ndarray,
    threshold: float,
    smoothing_window: int,
) -> tuple[pd.DataFrame, str]:
    df = base_eval_df.copy()
    df["error"] = errors
    df["threshold"] = float(threshold)
    df["y_pred"] = (df["error"].to_numpy(dtype=float) > float(threshold)).astype(np.int64)

    if smoothing_window <= 1:
        ae_pred_col = "y_pred"
        df["y_pred_smooth2"] = df["y_pred"].to_numpy(dtype=np.int64)
    else:
        ae_pred_col = "y_pred_smooth2"
        df["y_pred_smooth2"] = smooth_positive_runs(df["y_pred"].to_numpy(dtype=np.int64), min_run_length=smoothing_window)

    df["y_hybrid_strict"] = (
        (df[ae_pred_col].to_numpy(dtype=np.int64) == 1) &
        (df["rule_high_conf"].to_numpy(dtype=np.int64) == 1)
    ).astype(np.int64)
    df["y_hybrid_loose"] = (
        (df[ae_pred_col].to_numpy(dtype=np.int64) == 1) |
        (df["rule_high_conf"].to_numpy(dtype=np.int64) == 1)
    ).astype(np.int64)

    return df, ae_pred_col


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        dominated = (
            (df["Precision"] >= row["Precision"]) &
            (df["Recall"] >= row["Recall"]) &
            (
                (df["Precision"] > row["Precision"]) |
                (df["Recall"] > row["Recall"])
            )
        )
        dominated.iloc[idx] = False
        if not dominated.any():
            rows.append(row)

    pareto_df = pd.DataFrame(rows)
    if pareto_df.empty:
        return pareto_df

    return pareto_df.sort_values(
        ["F1-score", "Precision", "Recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main():
    print(f"Working directory: {ROOT}")
    print(f"Using device: {resolve_device()}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not REFERENCE_EVAL_PATH.exists():
        raise FileNotFoundError(f"Reference evaluation not found: {REFERENCE_EVAL_PATH}")

    print("Preparing validation/test sequences...")
    X_val_seq, X_test_seq, test_window_end = prepare_model_inputs(
        start_time=START_TIME,
        seq_len=SEQ_LEN,
        stride=STRIDE,
    )

    print("Loading reference evaluation with rule predictions...")
    reference_eval_df = pd.read_parquet(REFERENCE_EVAL_PATH).reset_index(drop=True)

    if len(reference_eval_df) != len(test_window_end):
        raise ValueError(
            f"Window length mismatch: reference={len(reference_eval_df)} test_windows={len(test_window_end)}"
        )

    ref_window_end = pd.to_datetime(reference_eval_df["window_end"])
    if not ref_window_end.equals(test_window_end):
        raise ValueError("window_end mismatch between reference evaluation and lightweight preprocessing")

    device = resolve_device()
    print("Loading trained model...")
    model = build_model().to(device)

    print("Computing timestep reconstruction errors for validation windows...")
    val_timestep_mse = get_timestep_mse(model, X_val_seq, device=device, batch_size=BATCH_SIZE)
    print("Computing timestep reconstruction errors for test windows...")
    test_timestep_mse = get_timestep_mse(model, X_test_seq, device=device, batch_size=BATCH_SIZE)

    score_cache = {}
    for cfg in SCORE_CONFIGS:
        key = (cfg["score_mode"], cfg["top_k"])
        score_cache[key] = {
            "val": compute_window_score(val_timestep_mse, mode=cfg["score_mode"], top_k=cfg["top_k"]),
            "test": compute_window_score(test_timestep_mse, mode=cfg["score_mode"], top_k=cfg["top_k"]),
        }

    reference_table = build_reference_table(reference_eval_df, label_col="label_dynamic")
    reference_table.to_csv(REFERENCE_TABLE_PATH, index=False)
    current_ae_ref = reference_table.loc[reference_table["Setting"] == "y_pred"].iloc[0]
    current_best_ref = reference_table.iloc[0]

    print("\nReference ranking")
    print(reference_table.to_string(index=False))

    y_true = reference_eval_df["label_dynamic"].to_numpy(dtype=np.int64)
    rows = []
    best_eval_df = None
    best_row = None

    print("\nSweeping score / threshold / smoothing / hybrid combinations...")
    for cfg in SCORE_CONFIGS:
        key = (cfg["score_mode"], cfg["top_k"])
        threshold_source = score_cache[key]["val"]
        test_errors = score_cache[key]["test"]

        for percentile in THRESHOLD_PERCENTILES:
            threshold = float(np.percentile(threshold_source, percentile))

            for smoothing_window in SMOOTHING_WINDOWS:
                candidate_df, ae_pred_col = build_candidate_eval_df(
                    base_eval_df=reference_eval_df,
                    errors=test_errors,
                    threshold=threshold,
                    smoothing_window=smoothing_window,
                )

                pred_map = {
                    "ae_only": ae_pred_col,
                    "strict_and_high_conf": "y_hybrid_strict",
                    "loose_or_high_conf": "y_hybrid_loose",
                }

                for hybrid_mode in HYBRID_MODES:
                    pred_col = pred_map[hybrid_mode]
                    y_pred = candidate_df[pred_col].to_numpy(dtype=np.int64)

                    row = {
                        "score_mode": cfg["score_mode"],
                        "top_k": cfg["top_k"] if cfg["score_mode"] == "topk" else np.nan,
                        "threshold_percentile": percentile,
                        "threshold": threshold,
                        "smoothing_window": smoothing_window,
                        "ae_pred_col": ae_pred_col,
                        "hybrid_mode": hybrid_mode,
                        "pred_col": pred_col,
                    }
                    row.update(compute_metrics(y_true, y_pred))
                    row["delta_precision_vs_current_y_pred"] = row["Precision"] - float(current_ae_ref["Precision"])
                    row["delta_recall_vs_current_y_pred"] = row["Recall"] - float(current_ae_ref["Recall"])
                    row["delta_f1_vs_current_y_pred"] = row["F1-score"] - float(current_ae_ref["F1-score"])
                    row["delta_precision_vs_current_best"] = row["Precision"] - float(current_best_ref["Precision"])
                    row["delta_recall_vs_current_best"] = row["Recall"] - float(current_best_ref["Recall"])
                    row["delta_f1_vs_current_best"] = row["F1-score"] - float(current_best_ref["F1-score"])
                    row["beats_current_y_pred_both"] = bool(
                        row["Precision"] > float(current_ae_ref["Precision"]) and
                        row["Recall"] > float(current_ae_ref["Recall"])
                    )
                    row["beats_current_best_f1"] = bool(row["F1-score"] > float(current_best_ref["F1-score"]))
                    row["beats_current_best_both"] = bool(
                        row["Precision"] > float(current_best_ref["Precision"]) and
                        row["Recall"] > float(current_best_ref["Recall"])
                    )
                    rows.append(row)

                    if best_row is None or (
                        row["F1-score"], row["Precision"], row["Recall"]
                    ) > (
                        best_row["F1-score"], best_row["Precision"], best_row["Recall"]
                    ):
                        best_row = row.copy()
                        best_eval_df = candidate_df.copy()

    result_df = pd.DataFrame(rows).sort_values(
        ["F1-score", "Precision", "Recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    pareto_df = pareto_front(result_df)

    result_df.to_csv(FULL_TABLE_PATH, index=False)
    pareto_df.to_csv(PARETO_TABLE_PATH, index=False)

    if best_eval_df is not None and best_row is not None:
        best_eval_df.to_parquet(BEST_EVAL_PATH, index=False)

    print("\nTop 15 candidate combinations")
    print(result_df.head(15).to_string(index=False))

    print("\nPareto front (Precision / Recall)")
    if pareto_df.empty:
        print("No pareto rows found.")
    else:
        print(pareto_df.to_string(index=False))

    print("\nBest candidate")
    if best_row is None:
        print("No candidate rows generated.")
    else:
        print(pd.DataFrame([best_row]).to_string(index=False))

    print("\nSaved files")
    print(FULL_TABLE_PATH)
    print(PARETO_TABLE_PATH)
    print(REFERENCE_TABLE_PATH)
    if best_eval_df is not None:
        print(BEST_EVAL_PATH)


if __name__ == "__main__":
    main()
