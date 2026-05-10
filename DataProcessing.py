from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# 計算超過一秒的時間間隔有哪些區段
def time_gaps(df):
    
    ts = pd.to_datetime(df["Timestamp"])
    diff = ts.diff().dt.total_seconds()
    gaps = df.loc[diff > 1, ["Timestamp"]].copy()
    gaps.insert(0, "start_time", ts.shift(1)[diff > 1].values)
    gaps["gap_seconds"] = diff[diff > 1].values
    all_gaps = gaps.reset_index(drop=True)
    # print(f"total gaps:", len(all_gaps))
    # display(all_gaps.sort_values("gap_seconds", ascending=False).head(10))
    
    return all_gaps

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len, stride, timestamps=None):
    
    X_seq, y_seq = [], []
    skipped_windows = 0
    ts_seconds = None

    # 將時間單位轉換成秒
    ts_seconds = pd.to_datetime(pd.Series(timestamps)).astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    # 依照 seq_len 的大小提取窗口
    for start in range(0, len(X)-seq_len+1, stride):
        end = start + seq_len

        # 檢查時間的連續性
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        X_window = X[start:end]
        y_window = y[start:end]

        X_seq.append(X_window)
        # 只要一個 window 內出現異常，就把整個 window 標成 1
        y_seq.append(int(y_window.max()))

    return (np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64), skipped_windows)


def Dataprocessing(start_time, drop_columns, SEQ_LEN, STRIDE):
    
    # 讀取資料
    df_normal = pd.read_parquet("../Dataset/SWaT_Dataset_Normal_v1.parquet")
    df_attack = pd.read_parquet("../Dataset/SWaT_Dataset_Attack_v1.parquet")
    print("\n------------------------- Original Data -------------------------")
    print(f"Normal Data = {df_normal.shape}")
    print(f"Attack Data = {df_attack.shape}")
    
    print("\n------------------------- Processing ... -------------------------")
    
    # 剩下的正常資料
    data = (df_normal['Timestamp'] >= start_time)
    df_normal = df_normal.loc[data]
    skip = data.shape[0] - len(df_normal)
    print(f"Normal data = {df_normal.shape}")
    print(f"Skip data = {skip}")
    
    # 合併所有資料
    df = pd.concat([df_normal, df_attack], axis=0, ignore_index=True)
    
    # 去掉多分類的標籤
    df = df.drop(drop_columns, axis=1)

    # 資料具有時序性所以不可以打亂
    train_set, other_normal = train_test_split(df_normal, train_size=0.8, shuffle=False) 
    val_set, tmp = train_test_split(other_normal, train_size=0.5, shuffle=False) 

    # 測試集保留剩餘正常資料，再加上全部異常資料
    test_set = pd.concat([tmp, df_attack], axis=0, ignore_index=True)

    # 檢查缺漏
    train_gaps = time_gaps(train_set)
    val_gaps = time_gaps(val_set)
    test_gaps = time_gaps(test_set)
    
    Label = test_set["Label"].value_counts()

    print(f"Number of total data = {df.shape}")
    print(f"train_set = {train_set.shape}")
    print(f"val_set = {val_set.shape}")
    print(f"test_set = {test_set.shape}")
    print(f"Test data has {int(Label[0])} normal data and {int(Label[1])} abnomaly data.")
    print(f"train gaps: {len(train_gaps)}")
    print(f"val gaps: {len(val_gaps)}")
    print(f"test gaps: {len(test_gaps)}")
    
    scaler = MinMaxScaler()

    feature_cols = df.columns.drop(['Timestamp', 'Label'])

    # 只用正常訓練資料
    train_x = scaler.fit_transform(train_set[feature_cols])
    val_x = scaler.transform(val_set[feature_cols])
    test_x = scaler.transform(test_set[feature_cols])

    train_y = train_set["Label"].to_numpy()
    val_y = val_set["Label"].to_numpy()
    test_y = test_set["Label"].to_numpy()

    # 提取時間
    train_timestamps = train_set["Timestamp"].to_numpy()
    val_timestamps = val_set["Timestamp"].to_numpy()
    test_timestamps = test_set["Timestamp"].to_numpy()

    X_train_seq, y_train_seq, train_skipped = make_sequences(train_x, train_y, SEQ_LEN, STRIDE, train_timestamps)
    X_val_seq, y_val_seq, val_skipped = make_sequences(val_x, val_y, SEQ_LEN, STRIDE, val_timestamps)
    X_test_seq, y_test_seq, test_skipped = make_sequences(test_x, test_y, SEQ_LEN, STRIDE, test_timestamps)
    
    print("\n------------------------- Final Result -------------------------")

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape, "skipped:", train_skipped)
    print("X_val_seq  :", X_val_seq.shape, "y_val_seq  :", y_val_seq.shape, "skipped:", val_skipped)
    print("X_test_seq :", X_test_seq.shape, "y_test_seq :", y_test_seq.shape, "skipped:", test_skipped)
    
    return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)
    
    

    
    
    
    
    