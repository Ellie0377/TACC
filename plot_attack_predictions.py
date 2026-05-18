from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    local_df = df.copy()
    for column in columns:
        if column in local_df.columns:
            local_df[column] = pd.to_datetime(local_df[column])
    return local_df


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _add_binary_strip(ax, time_values, binary_values, label: str, color: str, y_base: float) -> None:
    ax.step(time_values, y_base + binary_values.astype(float), where="post", color=color, linewidth=1.8, label=label)
    ax.fill_between(
        time_values,
        y_base,
        y_base + binary_values.astype(float),
        step="post",
        color=color,
        alpha=0.25,
    )


def _span_bounds(df: pd.DataFrame, start_col: str, end_col: str, mask: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans = []
    for start_idx, end_idx in _contiguous_spans(mask):
        spans.append((pd.Timestamp(df.loc[start_idx, start_col]), pd.Timestamp(df.loc[end_idx, end_col])))
    return spans


def _paint_spans(ax, spans: list[tuple[pd.Timestamp, pd.Timestamp]], color: str, alpha: float, label: str | None = None) -> None:
    for idx, (start_ts, end_ts) in enumerate(spans):
        ax.axvspan(start_ts, end_ts, color=color, alpha=alpha, label=label if idx == 0 else None, lw=0)


def plot_single_attack_prediction(
    evaluation_df: pd.DataFrame,
    segment_row,
    threshold: float | None = None,
    output_path: str | Path | None = None,
    time_col: str = "window_end",
    error_col: str = "error",
    pred_col: str = "y_pred",
    label_col: str = "label",
) -> plt.Figure | None:
    """Plot one attack segment with ground truth and model prediction."""

    if time_col not in evaluation_df.columns:
        raise ValueError(f"evaluation_df must contain '{time_col}'.")
    if error_col not in evaluation_df.columns:
        raise ValueError(f"evaluation_df must contain '{error_col}'.")
    if pred_col not in evaluation_df.columns:
        raise ValueError(f"evaluation_df must contain '{pred_col}'.")
    if label_col not in evaluation_df.columns:
        raise ValueError(f"evaluation_df must contain '{label_col}'.")

    local_df = _ensure_datetime_columns(evaluation_df, ["window_start", "window_end", "window_center", time_col])
    attack_group = int(segment_row.attack_segment_id)

    if "context_group" in local_df.columns:
        attack_df = local_df.loc[local_df["context_group"] == attack_group].copy()
    else:
        attack_df = local_df.loc[local_df.get("attack_segment_id", pd.Series(index=local_df.index, dtype=int)) == attack_group].copy()

    if attack_df.empty:
        print(f"[skip] {segment_row.Detailed_Label}: no windows found for context_group={attack_group}")
        return None

    attack_df = attack_df.sort_values(time_col).reset_index(drop=True)
    time_values = attack_df[time_col]
    error_values = attack_df[error_col].to_numpy(dtype=float)
    pred_values = attack_df[pred_col].to_numpy(dtype=int)
    label_values = attack_df[label_col].to_numpy(dtype=int)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.3]},
    )

    ax_err, ax_strip = axes

    ax_err.plot(time_values, error_values, color="#1f4e79", linewidth=1.5, label="reconstruction error")
    if threshold is not None:
        ax_err.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.3, label=f"threshold={threshold:.6f}")

    ax_err.axvspan(segment_row.start, segment_row.end, color="#f9d6d5", alpha=0.55, label="true attack interval")
    if hasattr(segment_row, "context_start") and hasattr(segment_row, "context_end"):
        ax_err.axvspan(segment_row.context_start, segment_row.context_end, color="#eef4fb", alpha=0.35, label="evaluation context")

    for start_idx, end_idx in _contiguous_spans(pred_values == 1):
        ax_err.axvspan(
            attack_df.loc[start_idx, "window_start"],
            attack_df.loc[end_idx, "window_end"],
            color="#f39c12",
            alpha=0.22,
        )

    ax_err.set_title(
        f"{segment_row.Detailed_Label}  |  "
        f"attack={segment_row.start:%Y-%m-%d %H:%M:%S} ~ {segment_row.end:%H:%M:%S}",
        fontsize=12,
    )
    ax_err.set_ylabel("Anomaly score")
    ax_err.grid(alpha=0.25, linestyle="--")
    ax_err.legend(loc="upper right", ncol=2)

    _add_binary_strip(ax_strip, time_values, label_values, "Ground truth", "#c0392b", y_base=1.15)
    _add_binary_strip(ax_strip, time_values, pred_values, "Prediction", "#f39c12", y_base=0.0)

    ax_strip.set_yticks([0.5, 1.65])
    ax_strip.set_yticklabels(["Prediction", "Ground truth"])
    ax_strip.set_ylim(-0.2, 2.4)
    ax_strip.grid(alpha=0.2, linestyle="--")
    ax_strip.legend(loc="upper right")
    ax_strip.set_xlabel("Timestamp")

    fp_windows = int(((pred_values == 1) & (label_values == 0)).sum())
    fn_windows = int(((pred_values == 0) & (label_values == 1)).sum())
    tp_windows = int(((pred_values == 1) & (label_values == 1)).sum())
    ax_strip.text(
        0.01,
        0.02,
        f"TP windows={tp_windows}  FP windows={fp_windows}  FN windows={fn_windows}",
        transform=ax_strip.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def plot_attack_signals_with_predictions(
    raw_attack_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    attack_name: str,
    signal_cols: list[str],
    output_path: str | Path | None = None,
    pad_before: str | pd.Timedelta = "5min",
    pad_after: str | pd.Timedelta = "5min",
    pred_col: str = "y_pred",
    label_col: str = "label",
    time_col: str = "window_end",
) -> plt.Figure | None:
    """
    Plot raw time-series values and overlay model-predicted normal/anomaly regions.

    Useful for answering:
    - model 是否只在攻擊後半段才抓到
    - 哪些區段是 attack 內的漏報
    - 哪些正常區段被誤判成 anomaly
    """

    required_eval_cols = {"context_group", "window_start", "window_end", pred_col, label_col, time_col}
    missing_eval = sorted(required_eval_cols - set(evaluation_df.columns))
    if missing_eval:
        raise ValueError(f"evaluation_df is missing required columns: {missing_eval}")

    required_seg_cols = {"attack_segment_id", "Detailed_Label", "start", "end"}
    missing_seg = sorted(required_seg_cols - set(segment_table.columns))
    if missing_seg:
        raise ValueError(f"segment_table is missing required columns: {missing_seg}")

    if "Timestamp" not in raw_attack_df.columns:
        raise ValueError("raw_attack_df must contain 'Timestamp'.")

    missing_signals = [col for col in signal_cols if col not in raw_attack_df.columns]
    if missing_signals:
        raise ValueError(f"raw_attack_df is missing signal columns: {missing_signals}")

    local_segments = _ensure_datetime_columns(segment_table, ["start", "end", "context_start", "context_end"])
    local_eval = _ensure_datetime_columns(evaluation_df, ["window_start", "window_end", "window_center", time_col])
    local_raw = _ensure_datetime_columns(raw_attack_df, ["Timestamp"])

    match = local_segments.loc[local_segments["Detailed_Label"] == attack_name].copy()
    if match.empty:
        raise ValueError(f"Attack '{attack_name}' was not found in segment_table.")

    segment_row = next(match.sort_values("attack_segment_id").itertuples(index=False))
    attack_group = int(segment_row.attack_segment_id)

    context_df = local_eval.loc[local_eval["context_group"] == attack_group].copy()
    if context_df.empty:
        raise ValueError(f"No evaluation windows found for attack '{attack_name}' (context_group={attack_group}).")

    pad_before = pd.Timedelta(pad_before)
    pad_after = pd.Timedelta(pad_after)
    plot_start = segment_row.start - pad_before
    plot_end = segment_row.end + pad_after

    if hasattr(segment_row, "context_start") and pd.notna(segment_row.context_start):
        plot_start = min(plot_start, segment_row.context_start)
    if hasattr(segment_row, "context_end") and pd.notna(segment_row.context_end):
        plot_end = max(plot_end, segment_row.context_end)

    signal_df = local_raw.loc[
        (local_raw["Timestamp"] >= plot_start) & (local_raw["Timestamp"] <= plot_end),
        ["Timestamp", *signal_cols],
    ].copy()
    if signal_df.empty:
        raise ValueError(f"No raw timestamps found for attack '{attack_name}' in the requested plotting range.")

    context_df = context_df.sort_values(time_col).reset_index(drop=True)
    pred_spans = _span_bounds(context_df, "window_start", "window_end", context_df[pred_col].to_numpy(dtype=int) == 1)
    normal_spans = _span_bounds(context_df, "window_start", "window_end", context_df[pred_col].to_numpy(dtype=int) == 0)
    miss_spans = _span_bounds(
        context_df,
        "window_start",
        "window_end",
        (context_df[label_col].to_numpy(dtype=int) == 1) & (context_df[pred_col].to_numpy(dtype=int) == 0),
    )

    fig, axes = plt.subplots(
        len(signal_cols) + 1,
        1,
        figsize=(16, 3.6 * len(signal_cols) + 2.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0] * len(signal_cols) + [1.1]},
    )
    if len(signal_cols) == 1:
        axes = np.asarray(axes, dtype=object)

    signal_axes = axes[:-1]
    strip_ax = axes[-1]

    for ax, signal_col in zip(signal_axes, signal_cols):
        ax.plot(signal_df["Timestamp"], signal_df[signal_col], color="#1f4e79", linewidth=1.4, label=signal_col)
        _paint_spans(ax, normal_spans, color="#d9f2d9", alpha=0.25, label="predicted normal")
        _paint_spans(ax, pred_spans, color="#f8c471", alpha=0.35, label="predicted anomaly")
        _paint_spans(ax, miss_spans, color="#e74c3c", alpha=0.18, label="missed attack part")
        ax.axvspan(segment_row.start, segment_row.end, color="#f5b7b1", alpha=0.22, label="true attack interval")
        ax.set_ylabel(signal_col)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="upper right", ncol=2, fontsize=9)

    pred_binary = context_df[pred_col].to_numpy(dtype=int)
    true_binary = context_df[label_col].to_numpy(dtype=int)
    time_values = context_df[time_col]
    _add_binary_strip(strip_ax, time_values, true_binary, "Ground truth", "#c0392b", y_base=1.15)
    _add_binary_strip(strip_ax, time_values, pred_binary, "Prediction", "#f39c12", y_base=0.0)
    strip_ax.set_yticks([0.5, 1.65])
    strip_ax.set_yticklabels(["Prediction", "Ground truth"])
    strip_ax.set_ylim(-0.2, 2.4)
    strip_ax.grid(alpha=0.2, linestyle="--")
    strip_ax.legend(loc="upper right")
    strip_ax.set_xlabel("Timestamp")

    tp_windows = int(((true_binary == 1) & (pred_binary == 1)).sum())
    fp_windows = int(((true_binary == 0) & (pred_binary == 1)).sum())
    fn_windows = int(((true_binary == 1) & (pred_binary == 0)).sum())
    hit_after_attack_start = None
    attack_hits = context_df.loc[(context_df[label_col] == 1) & (context_df[pred_col] == 1)]
    if not attack_hits.empty:
        hit_after_attack_start = attack_hits.iloc[0]["window_end"]

    title = (
        f"{attack_name} raw time-series with model prediction overlay\n"
        f"Attack: {segment_row.start:%Y-%m-%d %H:%M:%S} ~ {segment_row.end:%H:%M:%S}"
    )
    signal_axes[0].set_title(title, fontsize=13)

    note = (
        f"TP={tp_windows}  FP={fp_windows}  FN={fn_windows}"
        if hit_after_attack_start is None
        else f"TP={tp_windows}  FP={fp_windows}  FN={fn_windows}  first detected at {pd.Timestamp(hit_after_attack_start):%H:%M:%S}"
    )
    strip_ax.text(
        0.01,
        0.02,
        note,
        transform=strip_ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def plot_all_attack_predictions(
    evaluation_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    threshold: float | None = None,
    output_dir: str | Path = "results/attack_prediction_plots",
    attack_names: Iterable[str] | None = None,
    time_col: str = "window_end",
) -> list[Path]:
    """
    Plot every attack separately.

    Expected notebook objects:
    - evaluation_df: attack_window_df plus columns `error`, `y_pred`
    - segment_table: contains `attack_segment_id`, `Detailed_Label`, `start`, `end`,
      and optionally `context_start`, `context_end`
    - threshold: anomaly threshold used by the model
    """

    required_cols = {"attack_segment_id", "Detailed_Label", "start", "end"}
    missing = sorted(required_cols - set(segment_table.columns))
    if missing:
        raise ValueError(f"segment_table is missing required columns: {missing}")

    local_segment_table = _ensure_datetime_columns(segment_table, ["start", "end", "context_start", "context_end"])
    if attack_names is not None:
        wanted = set(attack_names)
        local_segment_table = local_segment_table.loc[local_segment_table["Detailed_Label"].isin(wanted)].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for segment_row in local_segment_table.sort_values("attack_segment_id").itertuples(index=False):
        file_name = f"{int(segment_row.attack_segment_id):02d}_{segment_row.Detailed_Label}.png"
        output_path = output_dir / file_name
        plot_single_attack_prediction(
            evaluation_df=evaluation_df,
            segment_row=segment_row,
            threshold=threshold,
            output_path=output_path,
            time_col=time_col,
        )
        saved_paths.append(output_path)

    print(f"Saved {len(saved_paths)} attack plots to: {output_dir}")
    return saved_paths


def build_attack_prediction_table(
    evaluation_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    pred_col: str = "y_pred",
    label_col: str = "label",
) -> pd.DataFrame:
    """Summarize each attack with detected / missed / fp / fn window counts."""

    required_eval_cols = {"context_group", pred_col, label_col}
    missing_eval = sorted(required_eval_cols - set(evaluation_df.columns))
    if missing_eval:
        raise ValueError(f"evaluation_df is missing required columns: {missing_eval}")

    required_seg_cols = {"attack_segment_id", "Detailed_Label", "start", "end"}
    missing_seg = sorted(required_seg_cols - set(segment_table.columns))
    if missing_seg:
        raise ValueError(f"segment_table is missing required columns: {missing_seg}")

    rows = []
    for segment_row in segment_table.sort_values("attack_segment_id").itertuples(index=False):
        attack_df = evaluation_df.loc[evaluation_df["context_group"] == int(segment_row.attack_segment_id)].copy()
        if attack_df.empty:
            continue

        y_true = attack_df[label_col].to_numpy(dtype=int)
        y_pred = attack_df[pred_col].to_numpy(dtype=int)
        rows.append(
            {
                "attack_segment_id": int(segment_row.attack_segment_id),
                "attack_name": segment_row.Detailed_Label,
                "start": segment_row.start,
                "end": segment_row.end,
                "detected": bool((y_pred[y_true == 1] == 1).any()) if (y_true == 1).any() else False,
                "n_windows": int(len(attack_df)),
                "n_attack_windows": int(y_true.sum()),
                "n_predicted_anomaly": int(y_pred.sum()),
                "tp_windows": int(((y_true == 1) & (y_pred == 1)).sum()),
                "fp_windows": int(((y_true == 0) & (y_pred == 1)).sum()),
                "fn_windows": int(((y_true == 1) & (y_pred == 0)).sum()),
            }
        )

    return pd.DataFrame(rows)
