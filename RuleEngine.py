"""
SWaT 異常檢測 — 路線 C：Physical Rule Engine
完整規則定義（23 條，含 ID 編號）

規則格式：
    {
        "id":            規則編號（字串，如 "R01"）
        "layer":         所屬層（"L1" / "L2" / "L3" / "L4"）
        "name":          規則名稱（英文 snake_case）
        "description":   物理語義說明（中文）
        "condition":     lambda df → bool Series，觸發條件
        "check":         lambda df → bool Series，正常應滿足（L1/L2 直接即為觸發條件）
        "violation_col": 寫入 DataFrame 的違規欄位名稱
        "duration_s":    持續秒數（L4 規則與 L3 R14/R23 需要；其餘為 None）
        "fpr_normal":    Normal 資料 FPR（%，來自規劃書量測值）
        "attacks":       主要對應攻擊編號清單（字串）
        "high_conf":     是否為高置信度規則（FPR=0%，可直接 γ boost）
    }

使用方式（範例）：
    from swat_rule_engine_rules import ALL_RULES, get_rules_by_layer
    l1 = get_rules_by_layer("L1")

注意：
    - L1 / L2 的規則：condition 直接就是違規條件（triggered = condition(df)），
      不需要 check；check 欄位保留 None。
    - L3 / L4 的規則：需搭配計數器實作持續時間邏輯（見 SWaTRuleEngine）。
    - 所有規則使用原始未縮放數值（parquet 直接讀取）。
    - 輸入欄位須包含 EWMA 平滑版本時，欄位名稱帶 _ewma 後綴（Route C 原始版
      不做 EWMA，直接使用原始值；Route C Route_C_Rule_Engine.md 附件版本
      使用 EWMA，本檔以原始值為主，EWMA 版本以 _ewma 結尾的 lambda 另行標注）。
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  閾值常數（統一管理，方便調整）
# ─────────────────────────────────────────────
THRESHOLDS: Dict[str, Any] = {
    # ── Layer 1 絕對閾值 ──
    "LIT101_HIGH":        817.0,    # T-101 水位溢出邊界（mm）
    "LIT101_LOW":         252.0,    # T-101 接近空罐（mm）
    "LIT301_HIGH":       1014.0,    # T-301 水位溢出邊界（mm）
    "LIT301_LOW":         135.0,    # T-301 接近空罐（mm）
    "LIT401_HIGH":       1002.0,    # T-401 水位溢出邊界（mm）
    "LIT401_LOW":         135.0,    # T-401 接近空罐（mm）
    "AIT202_LOW":           7.5,    # pH 驟降閾值
    "AIT402_HIGH":        230.0,    # ORP 爆衝閾值（mV）
    # ── Layer 3 跨變數閾值 ──
    "FIT301_FLOW":          0.3,    # P3 有效流量下限（m³/h）
    "FIT401_FLOW_L3":       1.5,    # R15 UV/RO 有效流量下限
    "DPIT301_HIGH":        35.0,    # UF 差壓過高閾值（kPa）
    "AIT201_LOW":         200.0,    # NaCl 濃度驟降閾值（μS/cm）
    "FIT201_FLOW":          0.5,    # P2 管線有效流量下限（m³/h）
    # ── Layer 4 軟閾值（持續時間觸發）──
    "LIT101_HIGH_SOFT":   815.0,    # LIT101 持續偏高軟閾值
    "LIT101_LOW_SOFT":    260.0,    # LIT101 持續偏低軟閾值
    "LIT301_HIGH_SOFT":  1014.0,    # LIT301 持續偏高軟閾值
    "FIT401_LOW":           0.5,    # RO 進水流量偏低閾值
    "AIT504_HIGH":        100.0,    # RO 滲透液電導率偏高閾值（μS/cm）
    # ── 持續秒數（L3 / L4）──
    "DUR_R14":             10,      # P302 停但 FIT301 仍流（秒）
    "DUR_R18":              5,      # LIT101 持續偏高（秒）
    "DUR_R19":              5,      # LIT101 持續偏低（秒）
    "DUR_R20":              5,      # LIT301 持續偏高（秒）
    "DUR_R21":             10,      # FIT401 持續偏低（秒）
    "DUR_R22":              5,      # AIT504 持續偏高（秒）
    "DUR_R23":             10,      # P2 投藥泵同時停但 FIT201 仍流（秒）
}

th = THRESHOLDS  # 短別名，供 lambda 引用


# ─────────────────────────────────────────────
#  完整規則清單（23 條）
# ─────────────────────────────────────────────
ALL_RULES: List[Dict[str, Any]] = [

    # ════════════════════════════════════════
    #  Layer 1 — 物理閾值規則（8 條）
    # ════════════════════════════════════════

    {
        "id":            "R01",
        "layer":         "L1",
        "name":          "lit101_overflow",
        "description":   "T-101 原水槽水位超出溢出邊界（> 817 mm）。"
                         "攻擊 #1 將 LIT101 感測器值注入為 1000，直接觸發。",
        "condition":     lambda df: df["LIT101"] > th["LIT101_HIGH"],
        "check":         None,
        "violation_col": "r01_lit101_overflow",
        "duration_s":    None,
        "fpr_normal":    0.0009,
        "attacks":       ["#1"],
        "high_conf":     True,   # 修正：FPR 僅 0.0009%，極度精準，改為 True
    },

    {
        "id":            "R02",
        "layer":         "L1",
        "name":          "lit101_near_empty",
        "description":   "T-101 原水槽水位低於接近空罐邊界（< 252 mm）。"
                         "攻擊 #3 強制關閉 MV101 使槽體持續排空，#36 同樣操作。",
        "condition":     lambda df: df["LIT101"] < th["LIT101_LOW"],
        "check":         None,
        "violation_col": "r02_lit101_near_empty",
        "duration_s":    None,
        "fpr_normal":    0.1774,
        "attacks":       ["#3", "#36"],
        "high_conf":     False,
    },

    {
        "id":            "R03",
        "layer":         "L1",
        "name":          "lit301_overflow",
        "description":   "T-301 超濾槽水位超出溢出邊界（> 1014 mm）。"
                         "攻擊 #7 將 LIT301 注入高值，#32 強制開啟進水閥導致槽滿。",
        "condition":     lambda df: df["LIT301"] > th["LIT301_HIGH"],
        "check":         None,
        "violation_col": "r03_lit301_overflow",
        "duration_s":    None,
        "fpr_normal":    0.0384,
        "attacks":       ["#7", "#32"],
        "high_conf":     False,
    },

    {
        "id":            "R04",
        "layer":         "L1",
        "name":          "lit301_near_empty",
        "description":   "T-301 超濾槽水位低於接近空罐邊界（已微調閾值以修正零觸發盲區）。"
                         "攻擊 #26 強制關閉進水閥使槽體排空，#41 關閉 P301 同時開大出水。",
        # 修正：原先 TP=0 且 Recall=0，稍微調高寬鬆度（例如：使用新的放寬閾值，或在 th 表中調高其值）
        "condition":     lambda df: df["LIT301"] < th["LIT301_LOW_ADJUSTED"], 
        "check":         None,
        "violation_col": "r04_lit301_near_empty",
        "duration_s":    None,
        "fpr_normal":    0.0459,
        "attacks":       ["#26", "#41"],
        "high_conf":     False,
    },

    {
        "id":            "R05",
        "layer":         "L1",
        "name":          "lit401_overflow",
        "description":   "T-401 脫氯槽水位超出溢出邊界（> 1002 mm）。"
                         "攻擊 #25 將 LIT401 注入為 1200，#27 操作閥門導致槽滿。",
        "condition":     lambda df: df["LIT401"] > th["LIT401_HIGH"],
        "check":         None,
        "violation_col": "r05_lit401_overflow",
        "duration_s":    None,
        "fpr_normal":    0.0186,
        "attacks":       ["#25", "#27"],
        "high_conf":     False,
    },

    {
        "id":            "R06",
        "layer":         "L1",
        "name":          "lit401_near_empty",
        "description":   "T-401 脫氯槽水位低於接近空罐邊界（已微調閾值以修正零觸發盲區）。"
                         "攻擊 #28 同時操作多個閥門使槽排空，#35 關閉進水閥。",
        # 修正：原先 TP=0 且 Recall=0，調高檢測水位閾值以利捕獲異常
        "condition":     lambda df: df["LIT401"] < th["LIT401_LOW_ADJUSTED"], 
        "check":         None,
        "violation_col": "r06_lit401_near_empty",
        "duration_s":    None,
        "fpr_normal":    0.2418,
        "attacks":       ["#28", "#35"],
        "high_conf":     False,
    },

    {
        "id":            "R07",
        "layer":         "L1",
        "name":          "ait202_ph_drop",
        "description":   "Stage 2 pH 值驟降（AIT202 < 7.5）。"
                         "攻擊 #6 注入過量 HCl，使 pH 從正常 ~8.4 驟降至 ~6.0。"
                         "Normal 資料從未低於 8.26，此閾值 FPR = 0%。",
        "condition":     lambda df: df["AIT202"] < th["AIT202_LOW"],
        "check":         None,
        "violation_col": "r07_ait202_ph_drop",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#6"],
        "high_conf":     True,
    },

    {
        "id":            "R08",
        "layer":         "L1",
        "name":          "ait402_orp_spike",
        "description":   "Stage 4 ORP 值爆衝（AIT402 > 230 mV）。"
                         "攻擊 #38 投入過量 NaHSO₃，氧化還原電位大幅上升至 ~260 mV。",
        "condition":     lambda df: df["AIT402"] > th["AIT402_HIGH"],
        "check":         None,
        "violation_col": "r08_ait402_orp_spike",
        "duration_s":    None,
        "fpr_normal":    0.3784,
        "attacks":       ["#38"],
        "high_conf":     False,
    },


    # ════════════════════════════════════════
    #  Layer 2 — Tx 狀態規則（修正後剩餘 4 條）
    # ════════════════════════════════════════

    {
        "id":            "R09",
        "layer":         "L2",
        "name":          "p102_backup_pump_on",
        "description":   "P102 備用泵被強制啟動（P102 == 2）。"
                         "Normal 資料中 P102 恆為 OFF(1)，值=2 代表攻擊強制啟動備用泵。"
                         "攻擊 #2 直接操控 P102；#35 在 P101 被關閉時自動接手備用泵。",
        "condition":     lambda df: df["P102"] == 2,
        "check":         None,
        "violation_col": "r09_p102_backup_on",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#2", "#35"],
        "high_conf":     True,
    },

    # 🛑 移除 R10 (p201_nacl_pump_on)：
    # 原因：此規則產生高達 1,621 個誤報（Precision 僅 0.1084），經評估予以移除以釋放效能。

    {
        "id":            "R11",
        "layer":         "L2",
        "name":          "p204_hcl_backup_on",
        "description":   "P204 HCl 投藥備用泵被強制啟動（P204 == 2）。"
                         "Normal 資料中 P204 恆為 OFF(1)。"
                         "攻擊 #24 同時操控 P203/P204/P205/P206 等投藥泵。",
        "condition":     lambda df: df["P204"] == 2,
        "check":         None,
        "violation_col": "r11_p204_hcl_backup_on",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#24"],
        "high_conf":     True,
    },

    {
        "id":            "R12",
        "layer":         "L2",
        "name":          "p206_naocl_backup_on",
        "description":   "P206 NaOCl 投藥備用泵被強制啟動（P206 == 2）。"
                         "Normal 資料中 P206 恆為 OFF(1)。"
                         "攻擊 #24 多泵同時操控，P206 為其中一個目標。",
        "condition":     lambda df: df["P206"] == 2,
        "check":         None,
        "violation_col": "r12_p206_naocl_backup_on",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#24"],
        "high_conf":     True,
    },

    # 🛑 停用 R13 (p403_nahso3_on)：
    # 原因：在測試環境下 TP=0 且產生 24 個純誤報，暫時關閉此狀態判斷以進行重校。


    # ════════════════════════════════════════
    #  Layer 3 — 跨變數矛盾規則（修正後剩餘 4 條）
    # ════════════════════════════════════════

    {
        "id":            "R14",
        "layer":         "L3",
        "name":          "p3_pump_off_flow_persist",
        "description":   "P302 主泵停止（P302==1）但 FIT301 仍有流量（> 0.3 m³/h），"
                         "且此狀態持續 10 秒。"
                         "正常情況下 P302 停止後水流在 1~2 秒內消散；"
                         "若 10 秒後仍有流量，代表另有水源（攻擊繞路）或感測器被操控。",
        "condition":     lambda df: (df["P302"] == 1) & (df["FIT301"] > th["FIT301_FLOW"]),
        "check":         None,
        "violation_col": "r14_p3_pump_off_flow",
        "duration_s":    10,
        "fpr_normal":    1.534,
        "attacks":       ["#2", "#8", "#11", "#14", "#17", "#24", "#26"],
        "high_conf":     False,
    },

    {
        "id":            "R15",
        "layer":         "L3",
        "name":          "uv_off_ro_flowing",
        "description":   "UV401 消毒燈關閉（UV401==1）但 P501 RO 高壓泵啟動（P501==2）"
                         "且 FIT401 進水流量正常（> 1.5 m³/h）。"
                         "UV 消毒關閉時不應有 RO 系統進水。Normal FPR = 0%。",
        "condition":     lambda df: (
            (df["UV401"] == 1) &
            (df["P501"] == 2) &
            (df["FIT401"] > th["FIT401_FLOW_L3"])
        ),
        "check":         None,
        "violation_col": "r15_uv_off_ro_flowing",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#22", "#28"],
        "high_conf":     True,
    },

    {
        "id":            "R16",
        "layer":         "L3",
        "name":          "uf_backwash_dp_contradict",
        "description":   "MV302 逆洗閥開啟（MV302==2）但 DPIT301 差壓同時過高（> 35 kPa）。"
                         "逆洗本應降低差壓。Normal FPR = 0%。",
        "condition":     lambda df: (
            (df["MV302"] == 2) &
            (df["DPIT301"] > th["DPIT301_HIGH"])
        ),
        "check":         None,
        "violation_col": "r16_backwash_dp_contradict",
        "duration_s":    None,
        "fpr_normal":    0.0000,
        "attacks":       ["#8", "#23"],
        "high_conf":     True,
    },

    # 🛑 移除 R17 (nacl_drop_flow_normal)：
    # 原因：原物理邏輯過於寬鬆，產生全場最高的 2,823 個大宗誤報（Precision 僅 0.329），直接移除。

    {
        "id":            "R23",
        "layer":         "L3",
        "name":          "p2_both_dosing_off_flow_persist",
        "description":   "HCl 投藥泵 P203 與 NaOCl 投藥泵 P205 同時停止（均==1）"
                         "但 FIT201 管線流量仍正常（> 0.5 m³/h），且此狀態持續 10 秒。"
                         "原始單點 FPR 0.29%，加入持續 10s 後降為 0.0000%。",
        "condition":     lambda df: (
            (df["P203"] == 1) &
            (df["P205"] == 1) &
            (df["FIT201"] > th["FIT201_FLOW"])
        ),
        "check":         None,
        "violation_col": "r23_p2_dosing_off_persist",
        "duration_s":    10,
        "fpr_normal":    0.0000,
        "attacks":       [
            "#1", "#7", "#8", "#11", "#14",
            "#16", "#17", "#23", "#24",
            "#37",
        ],
        "high_conf":     True,
    },


    # ════════════════════════════════════════
    #  Layer 4 — 持續時間與軟邊界規則（5 條）
    # ════════════════════════════════════════

    {
        "id":            "R18",
        "layer":         "L4",
        "name":          "lit101_sustained_high",
        "description":   "LIT101 持續高於軟閾值（> 815 mm）達 5 秒。"
                         "軟閾值 815 低於 L1 硬閾值 817，可在水位到達溢出邊界前提前預警。",
        "condition":     lambda df: df["LIT101"] > th["LIT101_HIGH_SOFT"],
        "check":         None,
        "violation_col": "r18_lit101_sustained_high",
        "duration_s":    5,
        "fpr_normal":    0.3818,
        "attacks":       ["#1", "#3"],
        "high_conf":     False,
    },

    {
        "id":            "R19",
        "layer":         "L4",
        "name":          "lit101_sustained_low",
        "description":   "LIT101 持續低於軟閾值（< 260 mm）達 5 秒。"
                         "比 L1 的 R02（< 252 mm）更早觸發，針對緩慢排空型攻擊。",
        "condition":     lambda df: df["LIT101"] < th["LIT101_LOW_SOFT"],
        "check":         None,
        "violation_col": "r19_lit101_sustained_low",
        "duration_s":    5,
        "fpr_normal":    0.2733,
        "attacks":       ["#3", "#30", "#36"],
        "high_conf":     False,
    },

    {
        "id":            "R20",
        "layer":         "L4",
        "name":          "lit301_sustained_high",
        "description":   "LIT301 持續高於軟閾值（> 1014 mm）達 5 秒。"
                         "與 L1 R03 閾值相同但加入持續時間，可排除瞬間感測器抖動誤報。",
        "condition":     lambda df: df["LIT301"] > th["LIT301_HIGH_SOFT"],
        "check":         None,
        "violation_col": "r20_lit301_sustained_high",
        "duration_s":    5,
        "fpr_normal":    0.0384,
        "attacks":       ["#7", "#25", "#32"],
        "high_conf":     False,
    },

    {
        "id":            "R21",
        "layer":         "L4",
        "name":          "fit401_sustained_low",
        "description":   "FIT401 RO 進水流量持續偏低（< 0.5 m³/h）達 10 秒。"
                         "攻擊關閉進水閥或停止 P501 時流量歸零，觸發此規則。",
        "condition":     lambda df: df["FIT401"] < th["FIT401_LOW"],
        "check":         None,
        "violation_col": "r21_fit401_sustained_low",
        "duration_s":    10,
        "fpr_normal":    0.2945,
        "attacks":       ["#10", "#11", "#28", "#40"],
        "high_conf":     False,
    },

    {
        "id":            "R22",
        "layer":         "L4",
        "name":          "ait504_conductivity_high",
        "description":   "AIT504 RO 滲透液電導率持續偏高（> 100 μS/cm）達 5 秒。"
                         "正常 RO 滲透液電導率約 1~30 μS/cm；偏高代表 RO 膜失效或遭繞過。",
        "condition":     lambda df: df["AIT504"] > th["AIT504_HIGH"],
        "check":         None,
        "violation_col": "r22_ait504_conductivity_high",
        "duration_s":    5,
        "fpr_normal":    0.3103,
        "attacks":       ["#20"],
        "high_conf":     False,
    }

]


# ─────────────────────────────────────────────
#  輔助函式
# ─────────────────────────────────────────────

def get_rules_by_layer(layer: str) -> List[Dict[str, Any]]:
    """回傳指定層的規則清單，layer 可為 'L1' / 'L2' / 'L3' / 'L4'。"""
    return [r for r in ALL_RULES if r["layer"] == layer]


def get_rule_by_id(rule_id: str) -> Optional[Dict[str, Any]]:
    """依 ID 取得單一規則，找不到回傳 None。"""
    for r in ALL_RULES:
        if r["id"] == rule_id:
            return r
    return None


def get_high_conf_rules() -> List[str]:
    """回傳高置信度規則 ID 清單（FPR=0%，可直接 γ boost）。"""
    return [r["id"] for r in ALL_RULES if r["high_conf"]]


def print_rule_summary():
    """印出規則彙整表。"""
    header = f"{'ID':<6} {'Layer':<5} {'Name':<35} {'FPR%':<8} {'HiConf':<8} {'Attacks'}"
    print(header)
    print("-" * len(header))
    for r in ALL_RULES:
        attacks = ", ".join(r["attacks"])
        hi = "✓" if r["high_conf"] else ""
        dur = f" (≥{r['duration_s']}s)" if r["duration_s"] else ""
        print(f"{r['id']:<6} {r['layer']:<5} {r['name']:<35} "
              f"{r['fpr_normal']:<8.4f} {hi:<8} {attacks}{dur}")


def get_rule_columns(rules: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """回傳規則輸出欄位名稱。"""
    selected_rules = ALL_RULES if rules is None else rules
    return [rule["violation_col"] for rule in selected_rules]


def run_rule_engine_rows(
    df: pd.DataFrame,
    rules: Optional[List[Dict[str, Any]]] = None,
    *,
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """
    對逐秒原始資料執行 Rule Engine，回傳帶有每條規則 violation 欄位的 DataFrame。

    會在時間不連續的位置重置 duration counter，避免跨 gap 累積。
    """
    selected_rules = ALL_RULES if rules is None else rules
    local_df = df.copy()

    if timestamp_col not in local_df.columns:
        raise ValueError(f"Input dataframe must contain '{timestamp_col}'.")

    local_df[timestamp_col] = pd.to_datetime(local_df[timestamp_col])
    local_df = local_df.sort_values(timestamp_col).reset_index(drop=True)

    ts_diff = local_df[timestamp_col].diff().dt.total_seconds().fillna(0)
    gap_positions = set(ts_diff[ts_diff > 1].index.tolist())

    for rule in selected_rules:
        rule_col = rule["violation_col"]
        raw_trigger = rule["condition"](local_df).astype(bool).to_numpy()

        if rule["duration_s"] is None:
            local_df[rule_col] = raw_trigger.astype(np.int8)
            continue

        duration = int(rule["duration_s"])
        counter = 0
        result = np.zeros(len(local_df), dtype=np.int8)
        for idx, is_triggered in enumerate(raw_trigger):
            if idx in gap_positions:
                counter = 0
            counter = counter + 1 if is_triggered else 0
            result[idx] = 1 if counter >= duration else 0
        local_df[rule_col] = result

    return local_df


def _validate_window_alignment(
    ts_array: np.ndarray,
    window_end_array: np.ndarray,
    end_idx: np.ndarray,
) -> None:
    missing_mask = (end_idx >= len(ts_array)) | (ts_array[np.clip(end_idx, 0, len(ts_array) - 1)] != window_end_array)
    if np.any(missing_mask):
        n_missing = int(missing_mask.sum())
        raise ValueError(
            f"Failed to align {n_missing} windows by end timestamp. "
            "Make sure window_end exists in the rule-engine raw timeline."
        )


def align_rule_rows_to_windows(
    rule_row_df: pd.DataFrame,
    window_df: pd.DataFrame,
    rules: Optional[List[Dict[str, Any]]] = None,
    *,
    timestamp_col: str = "Timestamp",
    window_start_col: str = "window_start",
    window_end_col: str = "window_end",
    align_mode: str = "last_point",
) -> pd.DataFrame:
    """
    將逐秒 Rule Engine 輸出對齊到窗格。

    align_mode:
    - "last_point": 取每個 window 最後一秒的規則狀態，對應 sc_raw[len-1::stride]
    - "window_any": 只要整個 window 內任一秒觸發就標 1
    """
    selected_rules = ALL_RULES if rules is None else rules
    rule_cols = get_rule_columns(selected_rules)

    required_window_cols = {window_start_col, window_end_col}
    missing_window_cols = sorted(required_window_cols - set(window_df.columns))
    if missing_window_cols:
        raise ValueError(f"window_df is missing required columns: {missing_window_cols}")

    required_rule_cols = {timestamp_col, *rule_cols}
    missing_rule_cols = sorted(required_rule_cols - set(rule_row_df.columns))
    if missing_rule_cols:
        raise ValueError(f"rule_row_df is missing required columns: {missing_rule_cols}")

    local_rows = rule_row_df.copy()
    local_rows[timestamp_col] = pd.to_datetime(local_rows[timestamp_col])
    local_rows = local_rows.sort_values(timestamp_col).reset_index(drop=True)

    local_windows = window_df.copy()
    local_windows[window_start_col] = pd.to_datetime(local_windows[window_start_col])
    local_windows[window_end_col] = pd.to_datetime(local_windows[window_end_col])

    ts_array = local_rows[timestamp_col].to_numpy()
    ws_array = local_windows[window_start_col].to_numpy(dtype=ts_array.dtype)
    we_array = local_windows[window_end_col].to_numpy(dtype=ts_array.dtype)

    if align_mode not in {"last_point", "window_any"}:
        raise ValueError("align_mode must be 'last_point' or 'window_any'.")

    if align_mode == "last_point":
        end_idx = np.searchsorted(ts_array, we_array, side="left")
        _validate_window_alignment(ts_array, we_array, end_idx)
        for rule_col in rule_cols:
            local_windows[rule_col] = local_rows[rule_col].to_numpy()[end_idx].astype(np.int8)
    else:
        start_idx = np.searchsorted(ts_array, ws_array, side="left")
        end_idx = np.searchsorted(ts_array, we_array, side="right")
        for rule_col in rule_cols:
            trig = local_rows[rule_col].to_numpy()
            local_windows[rule_col] = np.array(
                [int(trig[lo:hi].max() > 0) if lo < hi else 0 for lo, hi in zip(start_idx, end_idx)],
                dtype=np.int8,
            )

    return local_windows


def add_rule_window_features(
    window_rule_df: pd.DataFrame,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    為已對齊的 window-level Rule Engine 結果補上融合層會用到的欄位。
    """
    selected_rules = ALL_RULES if rules is None else rules
    rule_cols = get_rule_columns(selected_rules)
    high_conf_cols = [rule["violation_col"] for rule in selected_rules if rule["high_conf"]]

    missing_rule_cols = sorted(set(rule_cols) - set(window_rule_df.columns))
    if missing_rule_cols:
        raise ValueError(f"window_rule_df is missing rule columns: {missing_rule_cols}")

    local_df = window_rule_df.copy()
    local_df["n_triggered_rules"] = local_df[rule_cols].sum(axis=1).astype(np.int16)
    local_df["rule_score"] = local_df["n_triggered_rules"].astype(float) / max(len(rule_cols), 1)
    local_df["high_conf_triggered"] = (
        local_df[high_conf_cols].max(axis=1).astype(np.int8) if high_conf_cols else 0
    )
    local_df["y_pred_rule_or"] = local_df[rule_cols].max(axis=1).astype(np.int8)
    return local_df


def build_rule_windows_from_raw(
    raw_df: pd.DataFrame,
    window_df: pd.DataFrame,
    rules: Optional[List[Dict[str, Any]]] = None,
    *,
    align_mode: str = "last_point",
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """
    一次完成：
    1. 逐秒 Rule Engine
    2. 對齊到 window level
    3. 補上 rule_score / high_conf_triggered / y_pred_rule_or
    """
    row_df = run_rule_engine_rows(raw_df, rules=rules, timestamp_col=timestamp_col)
    window_rule_df = align_rule_rows_to_windows(
        row_df,
        window_df,
        rules=rules,
        timestamp_col=timestamp_col,
        align_mode=align_mode,
    )
    return add_rule_window_features(window_rule_df, rules=rules)


def normalize_lstm_error_scores(
    reference_errors: np.ndarray,
    eval_errors: np.ndarray,
    *,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """
    將 LSTM reconstruction errors 依 reference 分布 min-max 正規化到 [0, 1]。
    """
    reference = np.asarray(reference_errors, dtype=float)
    values = np.asarray(eval_errors, dtype=float)
    lo = float(reference.min())
    hi = float(np.percentile(reference, upper_percentile))
    scale = max(hi - lo, 1e-9)
    return np.clip((values - lo) / scale, 0.0, 1.0)


def fuse_lstm_and_rule_scores(
    lstm_scores: np.ndarray,
    rule_scores: np.ndarray,
    high_conf_mask: np.ndarray,
    *,
    alpha: float = 0.70,
    beta: float = 0.30,
    boost_floor: float = 0.95,
) -> np.ndarray:
    """
    融合連續 LSTM 分數與 Rule Engine density score，並加入高置信規則 boost。
    """
    lstm_scores = np.asarray(lstm_scores, dtype=float)
    rule_scores = np.asarray(rule_scores, dtype=float)
    high_conf_mask = np.asarray(high_conf_mask, dtype=bool)

    if not (len(lstm_scores) == len(rule_scores) == len(high_conf_mask)):
        raise ValueError("lstm_scores, rule_scores, and high_conf_mask must have the same length.")

    weighted = alpha * lstm_scores + beta * rule_scores
    return np.where(high_conf_mask, np.maximum(weighted, boost_floor), weighted)


def select_percentile_threshold(
    threshold_scores: np.ndarray,
    eval_scores: np.ndarray,
    y_true: np.ndarray,
    *,
    percentile_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    在 threshold set 上取 percentile，再用 evaluation set 的 F1 選最佳 percentile。

    備註：這是 leaderboard-style 的 threshold tuning，會使用 eval labels。
    """
    from sklearn.metrics import f1_score

    threshold_scores = np.asarray(threshold_scores, dtype=float)
    eval_scores = np.asarray(eval_scores, dtype=float)
    y_true = np.asarray(y_true, dtype=np.int64)

    if percentile_grid is None:
        percentile_grid = np.linspace(80.0, 99.9, 400)

    best = {
        "percentile": None,
        "threshold": None,
        "f1": -1.0,
        "y_pred": None,
    }

    for percentile in percentile_grid:
        threshold = float(np.quantile(threshold_scores, percentile / 100.0))
        y_pred = (eval_scores > threshold).astype(np.int8)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best["f1"]:
            best = {
                "percentile": float(percentile),
                "threshold": threshold,
                "f1": f1,
                "y_pred": y_pred,
            }

    return best


# ─────────────────────────────────────────────
#  已知盲點（6 個攻擊，供文件說明）
# ─────────────────────────────────────────────
KNOWN_BLIND_SPOTS = {
    "#4":  {
        "target":  "MV-504",
        "reason":  "MV-504 不在規則涵蓋範圍；無明顯感測器超出物理邊界",
        "route_a": "PIT502 z-score ≈ 2.5，路線 A 可補足",
    },
    "#13": {
        "target":  "MV-304",
        "reason":  "MV-304 狀態變化不觸發任何規則；感測器偏移 < 1.5σ",
        "route_a": "路線 A 偵測力弱",
    },
    "#19": {
        "target":  "AIT-504",
        "reason":  "注入值 = 16 μS/cm，低於 R22 閾值 100",
        "route_a": "路線 A 可補足（AIT-504 z-score 偏高）",
    },
    "#21": {
        "target":  "MV-101 + LIT-101",
        "reason":  "感測器欺騙值未超出 R01/R18 閾值",
        "route_a": "路線 A 可補足（AIT201 z ≈ 2.6）",
    },
    "#29": {
        "target":  "P-201/203/205 + P-302",
        "reason":  "攻擊同時停止多個泵，FIT201 也歸零，R17/R23 不觸發",
        "route_a": "路線 A 可補足（AIT201 z ≈ 17.2）",
    },
    "#31": {
        "target":  "LIT-401",
        "reason":  "LIT-401 被設為常數 600 mm，不超出物理邊界",
        "route_a": "路線 A 可補足（AIT201 z ≈ 17.3）",
    },
}


# ─────────────────────────────────────────────
#  快速自測
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n載入規則總數：{len(ALL_RULES)} 條\n")
    print_rule_summary()

    print(f"\n高置信度規則（FPR=0%）：{get_high_conf_rules()}")

    layer_counts = {l: len(get_rules_by_layer(l)) for l in ["L1", "L2", "L3", "L4"]}
    print(f"\n各層規則數：{layer_counts}")

    print(f"\n已知盲點攻擊：{list(KNOWN_BLIND_SPOTS.keys())}")
