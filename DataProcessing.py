from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from DataAnalysis.AttackInfo import (
    Sensors,
    Actuators,
    attack_info,
    attacks_time,
    stage_map,
    ATTACK_KEY_SENSOR_MAP,
    ATTACK_CRITICAL_ACTUATOR_MAP,
)

Sensors = [
    'FIT101', 'LIT101',
    'AIT201', 'AIT202', 'AIT203', 'FIT201',
    'DPIT301', 'FIT301', 'LIT301',
    'AIT401', 'AIT402', 'FIT401', 'LIT401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504',
    'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'PIT501', 'PIT502', 'PIT503',
    'FIT601',
]

Actuators = [
    'MV101', 'P101', 'P102',
    'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
    'P401', 'P402', 'P403', 'P404', 'UV401',
    'P501', 'P502',
    'P601', 'P602', 'P603',
]

NPI_ATTACK_IDS = {4, 5, 9, 12, 15, 18}
IGNORE_ATTACK4_IDS = {4}

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

# 建立滑動窗口
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


# 取視窗的最後一個資料的 Label 當作該視窗的 Label 
def make_sequence_end_labels(values: np.ndarray, seq_len, stride, timestamps=None):
    seq_values = []
    skipped_windows = 0

    ts_seconds = pd.to_datetime(pd.Series(timestamps)).astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    for start in range(0, len(values) - seq_len + 1, stride):
        end = start + seq_len
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        seq_values.append(int(values[end - 1]))

    return np.asarray(seq_values, dtype=np.int64), skipped_windows

# 只要一個資料的 Label＝1 則該視窗的 Label=1
def make_sequence_any_labels(values: np.ndarray, seq_len, stride, timestamps=None):
    seq_values = []
    skipped_windows = 0

    ts_seconds = pd.to_datetime(pd.Series(timestamps)).astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    for start in range(0, len(values) - seq_len + 1, stride):
        end = start + seq_len
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        seq_values.append(int(values[start:end].max()))

    return np.asarray(seq_values, dtype=np.int64), skipped_windows

# 過濾資料
def make_sequence_ignore_mask(values: np.ndarray, seq_len, stride, timestamps=None, ignore_ratio=0.5):
    seq_values = []
    skipped_windows = 0

    ts_seconds = pd.to_datetime(pd.Series(timestamps)).astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    for start in range(0, len(values) - seq_len + 1, stride):
        end = start + seq_len
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        # ignore 比例過高的 window 不列入該 evaluation setting
        seq_values.append(bool(np.mean(values[start:end]) < ignore_ratio))

    return np.asarray(seq_values, dtype=bool), skipped_windows

# 記錄每個視窗最後一筆資料的時間戳記
def make_sequence_end_timestamps(timestamps, seq_len, stride):
    seq_values = []
    skipped_windows = 0

    ts = pd.to_datetime(pd.Series(timestamps))
    ts_seconds = ts.astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    for start in range(0, len(ts) - seq_len + 1, stride):
        end = start + seq_len
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        seq_values.append(ts.iloc[end - 1])

    return np.asarray(seq_values), skipped_windows


# Sensor -> MinMax, Actuator -> devided by max value
def mixed_scale_features(train_df, val_df, test_df, feature_cols):

    train_scaled = pd.DataFrame(index=train_df.index, columns=feature_cols, dtype=np.float32)
    val_scaled = pd.DataFrame(index=val_df.index, columns=feature_cols, dtype=np.float32)
    test_scaled = pd.DataFrame(index=test_df.index, columns=feature_cols, dtype=np.float32)

    sensor_scaler = MinMaxScaler()
    train_scaled.loc[:, Sensors] = sensor_scaler.fit_transform(train_df[Sensors]).astype(np.float32)
    val_scaled.loc[:, Sensors] = sensor_scaler.transform(val_df[Sensors]).astype(np.float32)
    test_scaled.loc[:, Sensors] = sensor_scaler.transform(test_df[Sensors]).astype(np.float32)

    actuator_max = train_df[Actuators].max(axis=0).replace(0, 1.0).astype(np.float32)
    train_scaled.loc[:, Actuators] = train_df[Actuators].div(actuator_max, axis=1).astype(np.float32)
    val_scaled.loc[:, Actuators] = val_df[Actuators].div(actuator_max, axis=1).astype(np.float32)
    test_scaled.loc[:, Actuators] = test_df[Actuators].div(actuator_max, axis=1).astype(np.float32)

    return (
        train_scaled.to_numpy(dtype=np.float32),
        val_scaled.to_numpy(dtype=np.float32),
        test_scaled.to_numpy(dtype=np.float32),
        Sensors,
        Actuators,
    )


# Sensor 使用 EWMA 平滑後，再用 median ± k * MAD 建立 robust normal baseline
def compute_sensor_robust_baseline(df_normal_train: pd.DataFrame, sensor_cols, alpha=0.1, k=3.0):
    baseline = {}

    for col in sensor_cols:
        x = df_normal_train[col].ewm(alpha=alpha, adjust=False).mean()
        median = x.median()
        mad = (x - median).abs().median()
        robust_sigma = 1.4826 * mad

        if robust_sigma == 0 or np.isnan(robust_sigma):
            robust_sigma = x.std()

        if robust_sigma == 0 or np.isnan(robust_sigma):
            robust_sigma = 1e-6

        baseline[col] = {
            "median": median,
            "mad": mad,
            "robust_sigma": robust_sigma,
            "lower": median - k * robust_sigma,
            "upper": median + k * robust_sigma,
        }

    return pd.DataFrame(baseline).T


# Actuator 不做 EWMA，只記錄 normal training data 中觀察到的合法狀態
def compute_actuator_legal_states(df_normal_train: pd.DataFrame, actuator_cols):
    legal_states = {}

    for col in actuator_cols:
        legal_states[col] = set(df_normal_train[col].dropna().unique().tolist())

    return legal_states


# Actuator 狀態持續時間統計，後續可以用於分析「合法但持續過久」的狀態
def compute_actuator_duration_baseline(df_normal_train: pd.DataFrame, actuator_cols):
    duration_rows = []
    df_normal_train = df_normal_train.copy()
    df_normal_train["Timestamp"] = pd.to_datetime(df_normal_train["Timestamp"])

    for col in actuator_cols:
        state_group = (df_normal_train[col] != df_normal_train[col].shift()).cumsum()
        tmp = df_normal_train.groupby(state_group).agg(
            state=(col, "first"),
            start=("Timestamp", "first"),
            end=("Timestamp", "last"),
        )
        tmp["duration"] = (tmp["end"] - tmp["start"]).dt.total_seconds() + 1
        tmp["actuator"] = col

        for state, state_df in tmp.groupby("state"):
            duration_rows.append({
                "actuator": col,
                "state": state,
                "median_duration": state_df["duration"].median(),
                "q01_duration": state_df["duration"].quantile(0.01),
                "q99_duration": state_df["duration"].quantile(0.99),
            })

    return pd.DataFrame(duration_rows)


# 建立 attack segment table，排除 No Physical Impact Attack
def build_attack_segment_table():
    rows = []

    for attack_name, attack_start, attack_end in attacks_time:
        attack_id = int(attack_name.replace("Attack", ""))
        if attack_id in NPI_ATTACK_IDS:
            continue

        rows.append({
            "attack_name": attack_name,
            "attack_id": attack_id,
            "start": pd.Timestamp(attack_start),
            "end": pd.Timestamp(attack_end),
        })

    segment_table = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return segment_table


def mark_attack_ignore_mask(df: pd.DataFrame, attack_ids, col_name: str):
    """
    將指定 attack_id 的原始 attack period 標成 ignore mask。
    目前主要用於像 Attack4 這種資料集有標 attack、但在本研究設定中希望排除的事件。
    """
    if col_name not in df.columns:
        df[col_name] = 0

    if not attack_ids:
        return df

    for attack_name, attack_start, attack_end in attacks_time:
        attack_id = int(attack_name.replace("Attack", ""))
        if attack_id not in attack_ids:
            continue

        attack_start = pd.Timestamp(attack_start)
        attack_end = pd.Timestamp(attack_end)
        attack_mask = (
            (df["Timestamp"] >= attack_start) &
            (df["Timestamp"] <= attack_end)
        )
        df.loc[attack_mask, col_name] = 1

    return df


# 預設使用 AttackInfo.py 依據 attack point + stage 建出的 device maps。
# 若呼叫端想覆蓋預設行為，仍可自行傳入 key_sensor_map / critical_actuator_map。
def get_event_key_devices(row, key_sensor_map=None, critical_actuator_map=None):
    attack_id = int(row["attack_id"])
    key_sensor_map = ATTACK_KEY_SENSOR_MAP if key_sensor_map is None else key_sensor_map
    critical_actuator_map = (
        ATTACK_CRITICAL_ACTUATOR_MAP if critical_actuator_map is None else critical_actuator_map
    )

    if attack_id in key_sensor_map:
        key_sensors = key_sensor_map[attack_id]
    else:
        key_sensors = Sensors

    if attack_id in critical_actuator_map:
        critical_actuators = critical_actuator_map[attack_id]
    else:
        critical_actuators = Actuators

    return key_sensors, critical_actuators


# 判斷某個時間點的 key sensors 是否至少 80% 回到 robust normal range
def is_sensor_recovered(row, key_sensors, sensor_baseline, sensor_recovery_ratio=0.8):
    recovered = 0
    total = 0

    for col in key_sensors:
        ewma_col = f"{col}_ewma"
        value = row[ewma_col] if ewma_col in row.index else row[col]
        lower = sensor_baseline.loc[col, "lower"]
        upper = sensor_baseline.loc[col, "upper"]

        total += 1
        if lower <= value <= upper:
            recovered += 1

    if total == 0:
        return False

    return (recovered / total) >= sensor_recovery_ratio


# 判斷 critical actuators 是否全部回到 normal training data 中的合法狀態
def is_actuator_recovered(row, critical_actuators, actuator_legal_states):
    for col in critical_actuators:
        if row[col] not in actuator_legal_states[col]:
            return False

    return True


# 動態尋找 recovery end：Sensor 至少 80% 回正常範圍，且 Critical Actuator 全部合法，並連續維持數個 window
def find_dynamic_recovery_end(
    df_attack: pd.DataFrame,
    row,
    sensor_baseline,
    actuator_legal_states,
    key_sensors,
    critical_actuators,
    sensor_recovery_ratio=0.8,
    recovery_hold_windows=5,
    recovery_window_size=64,
    recovery_stride=3,
):
    search_df = df_attack[df_attack["Timestamp"] > row["end"]].copy()

    # Recovery 搜尋只到下一個 attack 開始前，避免把下一個 attack 的異常狀態當成本次 recovery
    if "next_start" in row.index and pd.notna(row["next_start"]):
        search_df = search_df[search_df["Timestamp"] < row["next_start"]]

    if len(search_df) < recovery_window_size:
        return row["end"]

    recovered_windows = 0

    for start in range(0, len(search_df) - recovery_window_size + 1, recovery_stride):
        end = start + recovery_window_size
        window = search_df.iloc[start:end]

        sensor_ok = []
        actuator_ok = []

        for _, current_row in window.iterrows():
            sensor_ok.append(
                is_sensor_recovered(current_row, key_sensors, sensor_baseline, sensor_recovery_ratio)
            )
            actuator_ok.append(
                is_actuator_recovered(current_row, critical_actuators, actuator_legal_states)
            )

        # 一個 recovery window 內大多數時間點都符合條件，才視為該 window recovery
        window_recovered = (np.mean(sensor_ok) >= 0.8) and (np.mean(actuator_ok) >= 0.8)

        if window_recovered:
            recovered_windows += 1
        else:
            recovered_windows = 0

        if recovered_windows >= recovery_hold_windows:
            return window["Timestamp"].iloc[-1]

    # 找不到 recovery end 時，回傳本次搜尋範圍最後時間；後續可用 recovery_unresolved 判斷是否排除
    return search_df["Timestamp"].iloc[-1]


# 建立 delay-aware / recovery-aware 評估用 label 與 ignore mask
def add_attack_buffer_labels_dynamic(
    df_attack: pd.DataFrame,
    df_normal_train: pd.DataFrame,
    lag_seconds: int = 30,
    sensor_alpha: float = 0.1,
    baseline_k: float = 3.0,
    sensor_recovery_ratio: float = 0.8,
    recovery_hold_windows: int = 5,
    recovery_window_size: int = 64,
    recovery_stride: int = 3,
    key_sensor_map=None,
    critical_actuator_map=None,
) -> pd.DataFrame:

    sensor_baseline = compute_sensor_robust_baseline(
        df_normal_train, Sensors, alpha=sensor_alpha, k=baseline_k
    )
    actuator_legal_states = compute_actuator_legal_states(df_normal_train, Actuators)
    actuator_duration_baseline = compute_actuator_duration_baseline(df_normal_train, Actuators)

    df_attack = df_attack.copy()
    df_attack["Timestamp"] = pd.to_datetime(df_attack["Timestamp"])

    # Sensor baseline / recovery 判斷使用 EWMA；模型輸入仍然只保留原始 51 個設備特徵
    for col in Sensors:
        df_attack[f"{col}_ewma"] = df_attack[col].ewm(alpha=sensor_alpha, adjust=False).mean()

    df_attack["label_lag"] = 0
    df_attack["label_recovery"] = 0
    df_attack["label_dynamic"] = df_attack["Label"].astype(int)
    df_attack["ignore_attack_period_only"] = 0
    df_attack["ignore_attack4"] = 0
    df_attack["ignore_buffer_10m"] = 0
    df_attack["ignore_buffer_30m"] = 0
    df_attack["ignore_buffer_60m"] = 0
    df_attack["recovery_unresolved"] = 0

    segment_table = build_attack_segment_table()
    segment_table["next_start"] = segment_table["start"].shift(-1)
    recovery_rows = []

    for _, row in segment_table.iterrows():
        key_sensors, critical_actuators = get_event_key_devices(
            row, key_sensor_map=key_sensor_map, critical_actuator_map=critical_actuator_map
        )

        recovery_end = find_dynamic_recovery_end(
            df_attack,
            row,
            sensor_baseline,
            actuator_legal_states,
            key_sensors,
            critical_actuators,
            sensor_recovery_ratio=sensor_recovery_ratio,
            recovery_hold_windows=recovery_hold_windows,
            recovery_window_size=recovery_window_size,
            recovery_stride=recovery_stride,
        )

        recovery_rows.append({
            "attack_name": row["attack_name"],
            "attack_id": row["attack_id"],
            "start": row["start"],
            "end": row["end"],
            "recovery_end": recovery_end,
            "recovery_seconds": (recovery_end - row["end"]).total_seconds(),
            "key_sensor_count": len(key_sensors),
            "critical_actuator_count": len(critical_actuators),
            "key_sensors": key_sensors,
            "critical_actuators": critical_actuators,
            "used_for_dynamic_label": True,
        })

        lag_end = row["start"] + pd.Timedelta(seconds=lag_seconds)
        lag_mask = (
            (df_attack["Timestamp"] >= row["start"]) &
            (df_attack["Timestamp"] < lag_end)
        )
        df_attack.loc[lag_mask, "label_lag"] = 1

        recovery_mask = (
            (df_attack["Timestamp"] > row["end"]) &
            (df_attack["Timestamp"] <= recovery_end) &
            (df_attack["Label"] == 0)
        )
        df_attack.loc[recovery_mask, "label_recovery"] = 1

        dynamic_mask = (
            (df_attack["Timestamp"] >= row["start"]) &
            (df_attack["Timestamp"] <= recovery_end)
        )
        df_attack.loc[dynamic_mask, "label_dynamic"] = 1

        for minutes in [10, 30, 60]:
            buffer_end = row["end"] + pd.Timedelta(minutes=minutes)
            buffer_mask = (
                (df_attack["Timestamp"] > row["end"]) &
                (df_attack["Timestamp"] <= buffer_end) &
                (df_attack["Label"] == 0)
            )
            df_attack.loc[buffer_mask, f"ignore_buffer_{minutes}m"] = 1

    first_attack_start = segment_table["start"].min()
    df_attack.loc[
        (df_attack["Timestamp"] >= first_attack_start) & (df_attack["Label"] == 0),
        "ignore_attack_period_only"
    ] = 1
    df_attack = mark_attack_ignore_mask(df_attack, IGNORE_ATTACK4_IDS, "ignore_attack4")

    # EWMA 欄位只供 recovery 判斷，不可進入模型輸入，避免增加特徵與 label/baseline leakage
    ewma_cols = [f"{col}_ewma" for col in Sensors]
    df_attack = df_attack.drop(columns=ewma_cols)

    df_attack.attrs["sensor_baseline"] = sensor_baseline
    df_attack.attrs["actuator_legal_states"] = actuator_legal_states
    df_attack.attrs["actuator_duration_baseline"] = actuator_duration_baseline
    df_attack.attrs["recovery_segments"] = pd.DataFrame(recovery_rows)

    return df_attack


# 建立 label_lag 和 label_recovery 的固定秒數標籤，保留作為舊版 buffer baseline
def add_attack_buffer_labels(df_attack: pd.DataFrame, lag_seconds: int = 30, recovery_seconds: int = 60):
    df_attack = df_attack.copy()
    df_attack["Timestamp"] = pd.to_datetime(df_attack["Timestamp"])
    df_attack["label_lag"] = 0
    df_attack["label_recovery"] = 0

    for attack_name, attack_start, attack_end in attacks_time:
        attack_id = int(attack_name.replace("Attack", ""))
        if attack_id in NPI_ATTACK_IDS:
            continue

        attack_start = pd.Timestamp(attack_start)
        attack_end = pd.Timestamp(attack_end)

        lag_end = attack_start + pd.Timedelta(seconds=lag_seconds)
        lag_mask = (df_attack["Timestamp"] >= attack_start) & (df_attack["Timestamp"] < lag_end)
        df_attack.loc[lag_mask, "label_lag"] = 1

        recovery_end = attack_end + pd.Timedelta(seconds=recovery_seconds)
        recovery_mask = (
            (df_attack["Timestamp"] > attack_end)
            & (df_attack["Timestamp"] <= recovery_end)
            & (df_attack["Label"] == 0)
        )
        df_attack.loc[recovery_mask, "label_recovery"] = 1

    return df_attack


def build_test_evaluation_metadata(test_set, SEQ_LEN, STRIDE):
    test_timestamps = test_set["Timestamp"].to_numpy()

    official_y, _ = make_sequence_any_labels(
        test_set["Label"].to_numpy(), SEQ_LEN, STRIDE, test_timestamps
    )
    dynamic_y, _ = make_sequence_any_labels(
        test_set["label_dynamic"].to_numpy(), SEQ_LEN, STRIDE, test_timestamps
    )
    lag_y, _ = make_sequence_any_labels(
        test_set["label_lag"].to_numpy(), SEQ_LEN, STRIDE, test_timestamps
    )
    recovery_y, _ = make_sequence_any_labels(
        test_set["label_recovery"].to_numpy(), SEQ_LEN, STRIDE, test_timestamps
    )
    end_timestamps, _ = make_sequence_end_timestamps(test_timestamps, SEQ_LEN, STRIDE)

    eval_masks = {}
    for col in [
        "ignore_attack_period_only",
        "ignore_attack4",
        "ignore_buffer_10m",
        "ignore_buffer_30m",
        "ignore_buffer_60m",
    ]:
        if col in test_set.columns:
            valid_mask, _ = make_sequence_ignore_mask(
                test_set[col].to_numpy(), SEQ_LEN, STRIDE, test_timestamps
            )
            eval_masks[col] = valid_mask

    eval_masks["official"] = np.ones_like(official_y, dtype=bool)
    eval_masks["dynamic_recovery"] = np.ones_like(dynamic_y, dtype=bool)

    return {
        "test_timestamps": end_timestamps,
        "test_y_official": official_y,
        "test_y_dynamic": dynamic_y,
        "test_y_lag": lag_y,
        "test_y_recovery_only": recovery_y,
        "eval_masks": eval_masks,
    }


def Dataprocessing(
    start_time,
    SEQ_LEN,
    STRIDE,
    return_metadata=False,
    sensor_alpha=0.1,
    baseline_k=3.0,
    sensor_recovery_ratio=0.8,
    recovery_hold_windows=5,
    key_sensor_map=None,
    critical_actuator_map=None,
):

    # 讀取資料
    df_normal = pd.read_parquet("../Dataset/SWaT_Dataset_Normal_v1.parquet")
    df_attack = pd.read_parquet("../Dataset/SWaT_Dataset_Attack_v1.parquet")
    df_normal["Timestamp"] = pd.to_datetime(df_normal["Timestamp"])
    df_attack["Timestamp"] = pd.to_datetime(df_attack["Timestamp"])

    df_normal_train = df_normal[df_normal["Timestamp"] >= start_time].copy()
    df_attack = add_attack_buffer_labels_dynamic(
        df_attack,
        df_normal_train,
        sensor_alpha=sensor_alpha,
        baseline_k=baseline_k,
        sensor_recovery_ratio=sensor_recovery_ratio,
        recovery_hold_windows=recovery_hold_windows,
        recovery_window_size=SEQ_LEN,
        recovery_stride=STRIDE,
        key_sensor_map=key_sensor_map,
        critical_actuator_map=critical_actuator_map,
    )

    recovery_segments = df_attack.attrs.get("recovery_segments", pd.DataFrame())
    sensor_baseline = df_attack.attrs.get("sensor_baseline", pd.DataFrame())
    actuator_duration_baseline = df_attack.attrs.get("actuator_duration_baseline", pd.DataFrame())

    df_normal = df_normal.copy()
    df_normal["label_lag"] = 0
    df_normal["label_recovery"] = 0
    df_normal["label_dynamic"] = df_normal["Label"].astype(int)
    df_normal["ignore_attack_period_only"] = 0
    df_normal["ignore_attack4"] = 0
    df_normal["ignore_buffer_10m"] = 0
    df_normal["ignore_buffer_30m"] = 0
    df_normal["ignore_buffer_60m"] = 0
    df_normal["recovery_unresolved"] = 0

    print("\n------------------------- Original Data -------------------------")
    print(f"Normal Data = {df_normal.shape}")
    print(f"Attack Data = {df_attack.shape}")
    
    print("\n------------------------- Processing ... -------------------------")
    
    # 剩下的正常資料，排除 startup / stabilization period 後才拿來訓練與建立 baseline
    data = (df_normal['Timestamp'] >= start_time)
    df_normal = df_normal.loc[data]
    skip = data.shape[0] - len(df_normal)
    print(f"Normal data = {df_normal.shape}")
    print(f"Skip data = {skip}")
    
    # 合併所有資料，只用來確認整體資料尺寸；模型輸入不會使用 label / ignore 欄位
    df = pd.concat([df_normal, df_attack], axis=0, ignore_index=True)
    
    # 去掉多分類的標籤
    df = df.drop(["Detailed_Label"], axis=1)

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

    # 嚴格限制模型輸入只包含 51 個設備特徵，避免 label_lag / label_recovery / ignore mask 造成 leakage
    feature_cols = Sensors + Actuators
    train_x, val_x, test_x, sensor_cols, actuator_cols = mixed_scale_features(
        train_set,
        val_set,
        test_set,
        feature_cols,
    )
    print(f"sensor features ({len(sensor_cols)}): {sensor_cols}")
    print(f"actuator features ({len(actuator_cols)}): {actuator_cols}")

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

    metadata = build_test_evaluation_metadata(test_set, SEQ_LEN, STRIDE)
    metadata["feature_cols"] = feature_cols
    metadata["sensor_cols"] = sensor_cols
    metadata["actuator_cols"] = actuator_cols
    metadata["sensor_baseline"] = sensor_baseline
    metadata["actuator_duration_baseline"] = actuator_duration_baseline
    metadata["recovery_segments"] = recovery_segments
    metadata["train_gaps"] = train_gaps
    metadata["val_gaps"] = val_gaps
    metadata["test_gaps"] = test_gaps
    attack_window_df_dict = {
        "window_end": metadata["test_timestamps"],
        "label": metadata["test_y_official"],
        "label_dynamic": metadata["test_y_dynamic"],
        "label_lag": metadata["test_y_lag"],
    }
    for mask_key in ["ignore_attack_period_only", "ignore_attack4", "ignore_buffer_10m", "ignore_buffer_30m", "ignore_buffer_60m"]:
        if mask_key in metadata["eval_masks"]:
            attack_window_df_dict[mask_key] = (~metadata["eval_masks"][mask_key]).astype(int)
    attack_window_df = pd.DataFrame(attack_window_df_dict)
    attack_window_df["window_start"] = pd.to_datetime(attack_window_df["window_end"]) - pd.Timedelta(seconds=SEQ_LEN - 1)
    metadata["attack_window_df"] = attack_window_df
    metadata["test_set"] = test_set.copy()
    metadata["actuator_legal_states"] = df_attack.attrs.get("actuator_legal_states", {})
    metadata["attack_info"] = attack_info
    metadata["attacks_time"] = attacks_time
    metadata["stage_map"] = stage_map
    metadata["attack_key_sensor_map"] = (
        ATTACK_KEY_SENSOR_MAP if key_sensor_map is None else key_sensor_map
    )
    metadata["attack_critical_actuator_map"] = (
        ATTACK_CRITICAL_ACTUATOR_MAP if critical_actuator_map is None else critical_actuator_map
    )
    metadata["npi_attack_ids"] = NPI_ATTACK_IDS
    metadata["ignore_attack4_ids"] = IGNORE_ATTACK4_IDS
    metadata["seq_len"] = SEQ_LEN
    metadata["stride"] = STRIDE
    
    print("\n------------------------- Recovery-aware Labels -------------------------")
    print(f"label_recovery rows = {int(test_set['label_recovery'].sum())}")
    print(f"label_dynamic rows = {int(test_set['label_dynamic'].sum())}")
    print(f"ignore Attack4 rows = {int(test_set['ignore_attack4'].sum())}")
    print(f"ignore 10m rows = {int(test_set['ignore_buffer_10m'].sum())}")
    print(f"ignore 30m rows = {int(test_set['ignore_buffer_30m'].sum())}")
    print(f"ignore 60m rows = {int(test_set['ignore_buffer_60m'].sum())}")
    print(f"recovery segments = {recovery_segments.shape}")

    print("\n------------------------- Final Result -------------------------")

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape, "skipped:", train_skipped)
    print("X_val_seq  :", X_val_seq.shape, "  y_val_seq  :", y_val_seq.shape, "  skipped:", val_skipped)
    print("X_test_seq :", X_test_seq.shape, " y_test_seq :", y_test_seq.shape, " skipped:", test_skipped)
    
    if return_metadata:
        return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, metadata)

    return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)
