from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


def load_rule_config(rule_path="Rule.json"):
    rule_path = Path(rule_path)
    with rule_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    rules = config.get("rules", [])
    thresholds = config.get("thresholds", {})
    high_conf_ids = set(config.get("high_conf_ids", []))

    # 若 JSON 沒有 high_conf_ids，就從每條 rule 的 high_conf 欄位取得。
    if len(high_conf_ids) == 0:
        high_conf_ids = {rule["id"] for rule in rules if bool(rule.get("high_conf", False))}

    rule_meta = pd.DataFrame(rules)
    return config, rules, thresholds, high_conf_ids, rule_meta


def _continuous_group_id(timestamps):
    ts = pd.to_datetime(pd.Series(timestamps))
    diff = ts.diff().dt.total_seconds().fillna(1)
    # diff != 1 代表資料中斷，要重置 duration 計算。
    return (diff != 1).cumsum()


def sustained_true(condition, timestamps, duration_s):
    """
    將 row-level condition 轉成「連續成立 duration_s 秒後才觸發」。
    SWaT 是 1 Hz 資料，因此 duration_s 約等於連續 row 數。
    若 timestamp 有 gap，會自動重置連續計數。
    """
    condition = pd.Series(condition).astype(bool).reset_index(drop=True)
    if duration_s is None or int(duration_s) <= 1:
        return condition.to_numpy(dtype=bool)

    timestamps = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    group_id = _continuous_group_id(timestamps)
    out = pd.Series(False, index=condition.index)
    window = int(duration_s)

    for _, idx in condition.groupby(group_id).groups.items():
        idx = list(idx)
        c = condition.iloc[idx].astype(int)
        out.iloc[idx] = c.rolling(window=window, min_periods=window).sum().ge(window).to_numpy()

    return out.to_numpy(dtype=bool)


def apply_rule_set(df: pd.DataFrame, rule_path="Rule.json"):
    """
    在 raw test_set rows 上套用 Rule.json。
    回傳每個 row 的 rule violation columns，以及 rule metadata。
    """
    config, rules, th, high_conf_ids, rule_meta = load_rule_config(rule_path)
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    out = pd.DataFrame(index=df.index)
    out["Timestamp"] = df["Timestamp"]

    # L1：sensor hard thresholds
    out["r01_lit101_overflow"] = df["LIT101"] > th["LIT101_HIGH"]
    out["r02_lit101_near_empty"] = df["LIT101"] < th["LIT101_LOW"]
    out["r03_lit301_overflow"] = df["LIT301"] > th["LIT301_HIGH"]
    out["r04_lit301_near_empty"] = df["LIT301"] < th["LIT301_LOW"]
    out["r05_lit401_overflow"] = df["LIT401"] > th["LIT401_HIGH"]
    out["r06_lit401_near_empty"] = df["LIT401"] < th["LIT401_LOW"]
    out["r07_ait202_ph_drop"] = df["AIT202"] < th["AIT202_LOW"]
    out["r08_ait402_orp_spike"] = df["AIT402"] > th["AIT402_HIGH"]

    # L2：actuator legal / unexpected states
    out["r09_p102_backup_on"] = df["P102"] == 2
    out["r11_p204_hcl_backup_on"] = df["P204"] == 2
    out["r12_p206_naocl_backup_on"] = df["P206"] == 2

    # L3：physical consistency rules
    r14_base = (df["P302"] == 1) & (df["FIT301"] > th["FIT301_FLOW"])
    out["r14_p3_pump_off_flow"] = sustained_true(r14_base, df["Timestamp"], th.get("DUR_R14", 10))

    out["r15_uv_off_ro_flowing"] = (
        (df["UV401"] == 1) &
        (df["P501"] == 2) &
        (df["FIT401"] > th["FIT401_FLOW_L3"])
    )

    out["r16_backwash_dp_contradict"] = (
        (df["MV302"] == 2) &
        (df["DPIT301"] > th["DPIT301_HIGH"])
    )

    r23_base = (
        (df["P203"] == 1) &
        (df["P205"] == 1) &
        (df["FIT201"] > th["FIT201_FLOW"])
    )
    out["r23_p2_dosing_off_persist"] = sustained_true(r23_base, df["Timestamp"], th.get("DUR_R23", 10))

    # L4：sustained soft rules
    out["r18_lit101_sustained_high"] = sustained_true(
        df["LIT101"] > th["LIT101_HIGH_SOFT"], df["Timestamp"], th.get("DUR_R18", 5)
    )
    out["r19_lit101_sustained_low"] = sustained_true(
        df["LIT101"] < th["LIT101_LOW_SOFT"], df["Timestamp"], th.get("DUR_R19", 5)
    )
    out["r20_lit301_sustained_high"] = sustained_true(
        df["LIT301"] > th["LIT301_HIGH_SOFT"], df["Timestamp"], th.get("DUR_R20", 5)
    )
    out["r21_fit401_sustained_low"] = sustained_true(
        df["FIT401"] < th["FIT401_LOW"], df["Timestamp"], th.get("DUR_R21", 10)
    )
    out["r22_ait504_conductivity_high"] = sustained_true(
        df["AIT504"] > th["AIT504_HIGH"], df["Timestamp"], th.get("DUR_R22", 5)
    )

    violation_cols = [rule["violation_col"] for rule in rules if rule.get("violation_col") in out.columns]
    high_conf_cols = [rule["violation_col"] for rule in rules if rule.get("id") in high_conf_ids and rule.get("violation_col") in out.columns]
    soft_cols = [col for col in violation_cols if col not in high_conf_cols]

    out[violation_cols] = out[violation_cols].astype(np.int64)
    out["rule_any"] = out[violation_cols].max(axis=1).astype(np.int64) if violation_cols else 0
    out["rule_high_conf"] = out[high_conf_cols].max(axis=1).astype(np.int64) if high_conf_cols else 0
    out["rule_soft"] = out[soft_cols].max(axis=1).astype(np.int64) if soft_cols else 0
    out["rule_soft_count"] = out[soft_cols].sum(axis=1).astype(np.int64) if soft_cols else 0
    out["rule_conservative"] = ((out["rule_high_conf"] == 1) | (out["rule_soft_count"] >= 2)).astype(np.int64)

    return out, rule_meta, violation_cols, high_conf_cols, soft_cols


def build_rule_window_df(test_set: pd.DataFrame, rule_row_df: pd.DataFrame, seq_len: int, stride: int):
    """
    使用與 LSTMAE 相同的 seq_len / stride / timestamp continuity rule，
    將 row-level rules 轉成 window-level rule predictions。
    """
    timestamps = pd.to_datetime(test_set["Timestamp"]).reset_index(drop=True)
    ts_seconds = timestamps.astype("int64") // 10**9
    ts_seconds = ts_seconds.to_numpy()

    rule_cols = [col for col in rule_row_df.columns if col != "Timestamp"]
    rows = []
    skipped_windows = 0

    rule_values = rule_row_df[rule_cols].reset_index(drop=True)

    for start in range(0, len(test_set) - seq_len + 1, stride):
        end = start + seq_len
        deltas = np.diff(ts_seconds[start:end])
        if not np.all(deltas == 1):
            skipped_windows += 1
            continue

        window = rule_values.iloc[start:end]
        row = {
            "window_start": timestamps.iloc[start],
            "window_end": timestamps.iloc[end - 1],
        }
        for col in rule_cols:
            row[col] = int(window[col].max())
        rows.append(row)

    return pd.DataFrame(rows), skipped_windows


def attach_rule_predictions(evaluation_df: pd.DataFrame, rule_window_df: pd.DataFrame):
    """
    將 rule window predictions 接到既有 LSTMAE evaluation_df。
    預設兩者由同一份 DataProcessing 產生，因此 window_end 應一一對齊。
    """
    eval_df = evaluation_df.copy().reset_index(drop=True)
    rule_window_df = rule_window_df.copy().reset_index(drop=True)

    if len(eval_df) != len(rule_window_df):
        raise ValueError(f"Length mismatch: evaluation_df={len(eval_df)}, rule_window_df={len(rule_window_df)}")

    eval_times = pd.to_datetime(eval_df["window_end"]).reset_index(drop=True)
    rule_times = pd.to_datetime(rule_window_df["window_end"]).reset_index(drop=True)
    if not eval_times.equals(rule_times):
        mismatch = np.where(eval_times.to_numpy() != rule_times.to_numpy())[0]
        first = int(mismatch[0]) if len(mismatch) else None
        raise ValueError(f"window_end mismatch at index {first}")

    add_cols = [col for col in rule_window_df.columns if col not in ["window_start", "window_end"]]
    for col in add_cols:
        eval_df[col] = rule_window_df[col].to_numpy(dtype=np.int64)

    return eval_df


def compute_binary_metrics(eval_df, label_col="label_dynamic", pred_col="rule_any", score_col=None, ignore_col=None):
    if ignore_col is None:
        mask = np.ones(len(eval_df), dtype=bool)
    else:
        mask = eval_df[ignore_col].to_numpy(dtype=int) == 0

    y_true = eval_df.loc[mask, label_col].to_numpy(dtype=np.int64)
    y_pred = eval_df.loc[mask, pred_col].to_numpy(dtype=np.int64)

    if score_col is None:
        scores = y_pred.astype(float)
    else:
        scores = eval_df.loc[mask, score_col].to_numpy(dtype=float)

    return {
        "Evaluated Windows": int(mask.sum()),
        "Ignored Windows": int((~mask).sum()),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-PR": average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
        "Positive Windows": int(y_pred.sum()),
    }


def evaluate_rule_sets(eval_df: pd.DataFrame, label_col="label_dynamic", ignore_col=None):
    rows = []
    aggregate_cols = ["rule_any", "rule_high_conf", "rule_soft", "rule_conservative"]
    for pred_col in aggregate_cols:
        if pred_col not in eval_df.columns:
            continue
        row = {"Setting": pred_col}
        row.update(compute_binary_metrics(eval_df, label_col=label_col, pred_col=pred_col, ignore_col=ignore_col))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_individual_rules(eval_df: pd.DataFrame, rule_meta: pd.DataFrame, label_col="label_dynamic", ignore_col=None):
    rows = []
    for _, rule in rule_meta.iterrows():
        col = rule.get("violation_col")
        if col not in eval_df.columns:
            continue
        row = {
            "Rule ID": rule.get("id"),
            "Rule Name": rule.get("name"),
            "Layer": rule.get("layer"),
            "High Confidence": bool(rule.get("high_conf", False)),
            "Original FPR Normal": rule.get("fpr_normal", np.nan),
            "Target Attacks": ", ".join(rule.get("attacks", [])) if isinstance(rule.get("attacks", []), list) else rule.get("attacks"),
        }
        row.update(compute_binary_metrics(eval_df, label_col=label_col, pred_col=col, ignore_col=ignore_col))
        rows.append(row)

    return pd.DataFrame(rows)


def add_hybrid_predictions(eval_df: pd.DataFrame, ae_pred_col="y_pred"):
    """
    建立幾種常用 hybrid decision：
    - hybrid_or_high_conf: LSTMAE OR high-confidence rule
    - hybrid_or_any_rule: LSTMAE OR any rule（通常 recall 高，但 precision 可能下降）
    - hybrid_conservative: high-confidence rule OR (LSTMAE AND soft rule)
    - hybrid_adaptive_balanced: 對 no-rule / soft-rule / strong-rule 採不同 error threshold
    - hybrid_adaptive_f1: 偏向提升整體 F1 的 adaptive threshold 版本
    """
    df = eval_df.copy()
    df["hybrid_or_high_conf"] = ((df[ae_pred_col] == 1) | (df["rule_high_conf"] == 1)).astype(np.int64)
    df["hybrid_or_any_rule"] = ((df[ae_pred_col] == 1) | (df["rule_any"] == 1)).astype(np.int64)
    df["hybrid_conservative"] = (
        (df["rule_high_conf"] == 1) |
        ((df[ae_pred_col] == 1) & (df["rule_soft"] == 1))
    ).astype(np.int64)

    if {"error", "threshold", "rule_high_conf", "rule_soft_count"}.issubset(df.columns):
        df = add_adaptive_hybrid_prediction(
            df,
            pred_col="hybrid_adaptive_balanced",
            no_rule_threshold_mul=1.15,
            soft_rule_threshold_mul=1.00,
            multi_soft_threshold_mul=0.65,
            direct_high_conf=True,
        )
        df = add_adaptive_hybrid_prediction(
            df,
            pred_col="hybrid_adaptive_f1",
            no_rule_threshold_mul=1.40,
            soft_rule_threshold_mul=1.00,
            multi_soft_threshold_mul=0.65,
            direct_high_conf=True,
        )
    return df


def evaluate_hybrid_sets(eval_df: pd.DataFrame, label_col="label_dynamic", ignore_col=None):
    rows = []
    for pred_col in [
        "y_pred",
        "hybrid_or_high_conf",
        "hybrid_or_any_rule",
        "hybrid_conservative",
        "hybrid_adaptive_balanced",
        "hybrid_adaptive_f1",
    ]:
        if pred_col not in eval_df.columns:
            continue
        row = {"Setting": pred_col}
        score_col = "error" if pred_col == "y_pred" and "error" in eval_df.columns else None
        row.update(compute_binary_metrics(eval_df, label_col=label_col, pred_col=pred_col, score_col=score_col, ignore_col=ignore_col))
        rows.append(row)
    return pd.DataFrame(rows)


def add_adaptive_hybrid_prediction(
    eval_df: pd.DataFrame,
    pred_col="hybrid_adaptive_balanced",
    error_col="error",
    threshold_col="threshold",
    high_conf_col="rule_high_conf",
    soft_count_col="rule_soft_count",
    no_rule_threshold_mul=1.15,
    soft_rule_threshold_mul=1.00,
    multi_soft_threshold_mul=0.65,
    direct_high_conf=True,
):
    """
    Adaptive threshold hybrid:
    - 沒有規則支撐時，提高 AE threshold 以壓低 false alarm
    - 有 soft rule 時，維持或降低 threshold 以保留 recall
    - soft rule >= 2 時，進一步降低 threshold
    - high-confidence rule 可直接觸發異常

    這種設計比單純 OR/AND 更適合「precision 與 recall 一起往上推」。
    """
    required_cols = {error_col, threshold_col, high_conf_col, soft_count_col}
    missing = required_cols.difference(eval_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for adaptive hybrid: {sorted(missing)}")

    df = eval_df.copy()
    error = df[error_col].to_numpy(dtype=float)
    base_threshold = df[threshold_col].to_numpy(dtype=float)
    high_conf = df[high_conf_col].to_numpy(dtype=np.int64)
    soft_count = df[soft_count_col].to_numpy(dtype=np.int64)

    adaptive_threshold = base_threshold * float(no_rule_threshold_mul)
    adaptive_threshold = np.where(
        soft_count >= 1,
        base_threshold * float(soft_rule_threshold_mul),
        adaptive_threshold,
    )
    adaptive_threshold = np.where(
        soft_count >= 2,
        base_threshold * float(multi_soft_threshold_mul),
        adaptive_threshold,
    )

    pred = error > adaptive_threshold
    if direct_high_conf:
        pred = pred | (high_conf == 1)

    df[f"{pred_col}_threshold"] = adaptive_threshold.astype(float)
    df[pred_col] = pred.astype(np.int64)
    return df


def search_adaptive_hybrid_configs(
    eval_df: pd.DataFrame,
    label_col="label_dynamic",
    ignore_col=None,
    no_rule_threshold_grid=(1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40),
    soft_rule_threshold_grid=(0.85, 0.90, 0.95, 1.00),
    multi_soft_threshold_grid=(0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    direct_high_conf_options=(True,),
):
    """
    針對 adaptive hybrid 做 grid search。
    預設用 dynamic recovery-aware label 評估，方便直接比較是否優於 baseline AE。
    """
    rows = []
    for direct_high_conf in direct_high_conf_options:
        for no_rule_mul in no_rule_threshold_grid:
            for soft_mul in soft_rule_threshold_grid:
                for multi_soft_mul in multi_soft_threshold_grid:
                    tmp = add_adaptive_hybrid_prediction(
                        eval_df,
                        pred_col="hybrid_adaptive_search",
                        no_rule_threshold_mul=no_rule_mul,
                        soft_rule_threshold_mul=soft_mul,
                        multi_soft_threshold_mul=multi_soft_mul,
                        direct_high_conf=direct_high_conf,
                    )
                    metric_row = compute_binary_metrics(
                        tmp,
                        label_col=label_col,
                        pred_col="hybrid_adaptive_search",
                        ignore_col=ignore_col,
                    )
                    metric_row.update({
                        "Direct High Conf": bool(direct_high_conf),
                        "No Rule Mul": float(no_rule_mul),
                        "Soft Rule Mul": float(soft_mul),
                        "Multi Soft Mul": float(multi_soft_mul),
                    })
                    rows.append(metric_row)

    return pd.DataFrame(rows).sort_values(
        ["F1-score", "Precision", "Recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
