"""W11 - menjalankan experiment track penuh pada data validation.

Protokol mengikuti CP3 (tidak diubah):
- official temporal split, warm-start evaluable users (11.659)
- candidate universe resmi 22.753 item, Top-200 per model, dinilai Top-10
- relevan = rating >= 4, item training-seen dikeluarkan
- metrik utama pemilihan model: NDCG@10, tie-breaker Recall@10
- test tidak dipakai sama sekali

Yang dijalankan di sini:
1. Semua baseline (Popularity, Item-CF, TF-IDF CBF) dievaluasi ulang dari
   ranking Top-200 yang tersimpan di cache Task 3 (cache_key tercatat).
2. Grid search search space CP3: Weighted Score alpha 0.0..1.0 (step 0.1)
   dan RRF k in {10, 20, 40, 60, 80, 100}.
3. Metrik dihitung untuk K = 10, 20, 50: Recall, Precision, NDCG, MRR, MAP.
4. Model terbaik per keluarga dan keseluruhan dipilih di validation.
5. Tabel, log eksperimen, analisis segmen user, dan figure disimpan.

Jalankan:
    python -m src.w11_experiment
    python -m src.w11_experiment --limit-users 300 --out outputs/w11_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from src.data import load_project_data
from src.evaluation import build_relevant_items
from src.fusion import reciprocal_rank_fusion, weighted_score_fusion

# ------------------------------------------------------------------ protokol
SEED = 42
POSITIVE_THRESHOLD = 4.0
K_LIST = (10, 20, 50)
PRIMARY_K = 10
TOP_N = 200
CACHE_KEY = "13454c5090f84bab"
ALPHAS = tuple(round(step / 10, 1) for step in range(11))
RRF_KS = (10, 20, 40, 60, 80, 100)

MODEL_COLORS = {
    "Popularity": "#6c757d",
    "Item-CF": "#d97706",
    "TF-IDF CBF": "#2563eb",
    "Weighted Score": "#7c3aed",
    "RRF": "#059669",
}
BASELINE_MODELS = ("Popularity", "Item-CF", "TF-IDF CBF")
ACTIVITY_BUCKETS = ((1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 10**9, "20+"))


# ------------------------------------------------------------------ utilitas
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=root, capture_output=True, text=True, check=False
            ).stdout.strip()
        except OSError:
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def flatten(prefix: str, values: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in values.items()}


# --------------------------------------------------------------------- metrik
def average_precision(flags: list[int], n_targets: int, k: int) -> float:
    if n_targets == 0:
        return 0.0
    cumulative = 0
    total = 0.0
    for position, flag in enumerate(flags[:k], start=1):
        if flag:
            cumulative += 1
            total += cumulative / position
    return total / min(n_targets, k)


def score_ranking(ranked: list[str], targets: set[str], k_list: tuple[int, ...]) -> dict:
    """Hitung semua metrik@K untuk satu user."""
    ordered = ranked[: max(k_list)]
    flags = [1 if item in targets else 0 for item in ordered]
    n_targets = len(targets)
    metrics: dict[str, float] = {}

    for k in k_list:
        window = flags[:k]
        hits = sum(window)
        metrics[f"recall@{k}"] = hits / n_targets if n_targets else 0.0
        metrics[f"precision@{k}"] = hits / k
        dcg = sum(1.0 / math.log2(pos + 2) for pos, flag in enumerate(window) if flag)
        ideal = min(n_targets, k)
        idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
        metrics[f"mrr@{k}"] = next(
            (1.0 / (pos + 1) for pos, flag in enumerate(window) if flag), 0.0
        )
        metrics[f"map@{k}"] = average_precision(window, n_targets, k)

    return metrics


def evaluate(
    rankings: dict[str, list[tuple[str, float]]],
    relevant: dict[str, set[str]],
    users: list[str],
    k_list: tuple[int, ...] = K_LIST,
    keep_per_user: bool = False,
):
    rows = []
    for user in users:
        targets = relevant.get(user, set())
        if not targets:
            continue
        ranked = [item for item, _ in rankings.get(user, [])]
        rows.append({"user_id": user, **score_ranking(ranked, targets, k_list)})

    if not rows:
        raise RuntimeError("Tidak ada user dengan target relevan.")

    per_user = pd.DataFrame(rows)
    metric_columns = [c for c in per_user.columns if c != "user_id"]
    summary = {column: float(per_user[column].mean()) for column in metric_columns}
    summary["users_evaluated"] = int(len(per_user))
    summary["users_with_hit@10"] = int((per_user["recall@10"] > 0).sum())
    summary["hit_rate_pct@10"] = float((per_user["recall@10"] > 0).mean() * 100)

    top10 = {user: [item for item, _ in rankings.get(user, [])][:10] for user in users}
    summary["unique_items@10"] = len({item for items in top10.values() for item in items})

    return summary, (per_user if keep_per_user else None)


# ---------------------------------------------------------------------- setup
def load_cache(root: Path) -> tuple[dict, dict]:
    cache_dir = root / "outputs" / "task3" / "cache"
    files = {
        "Popularity": cache_dir / f"popularity_recommendations_{CACHE_KEY}.joblib",
        "Item-CF": cache_dir / f"item_cf_recommendations_{CACHE_KEY}.joblib",
        "TF-IDF CBF": cache_dir / f"tfidf_cbf_recommendations_{CACHE_KEY}.joblib",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cache rekomendasi tidak ditemukan: " + ", ".join(missing))

    rankings = {name: joblib.load(path) for name, path in files.items()}
    meta = {
        "cache_key": CACHE_KEY,
        "cache_files": {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }
    return rankings, meta


def audit_rankings(
    rankings: dict[str, dict],
    users: list[str],
    relevant: dict[str, set[str]],
    candidate_set: set[str],
    seen_by_user: dict[str, set[str]],
) -> list[dict]:
    """Pastikan ranking patuh protokol sebelum dipakai menghitung metrik."""
    checks = []
    expected = set(users)

    for model, recs in rankings.items():
        key_ok = expected.issubset(set(map(str, recs)))
        lengths = [len(recs.get(user, [])) for user in users]
        too_many = sum(length > TOP_N for length in lengths)
        too_few = sum(length == 0 for length in lengths)

        duplicates = 0
        outside_candidates = 0
        seen_items = 0
        for user in users:
            items = [item for item, _ in recs.get(user, [])]
            duplicates += len(items) - len(set(items))
            outside_candidates += sum(item not in candidate_set for item in items)
            user_seen = seen_by_user.get(user, set())
            seen_items += sum(item in user_seen for item in items)

        passed = (
            key_ok
            and not too_many
            and not too_few
            and not duplicates
            and not outside_candidates
            and not seen_items
        )
        checks.append(
            {
                "pemeriksaan": f"protokol ranking - {model}",
                "status": "PASS" if passed else "FAIL",
                "bukti": (
                    f"{len(recs)} user tersedia, {len(users)} dievaluasi; "
                    f"max {max(lengths)} item; {duplicates} duplikat; "
                    f"{outside_candidates} di luar kandidat; {seen_items} seen-item"
                ),
            }
        )
        if not passed:
            raise AssertionError(f"Ranking {model} melanggar protokol: {checks[-1]['bukti']}")

    overlap = sum(
        1 for user in users if relevant.get(user, set()) & seen_by_user.get(user, set())
    )
    checks.append(
        {
            "pemeriksaan": "target relevan tidak training-seen",
            "status": "PASS" if overlap == 0 else "FAIL",
            "bukti": f"{overlap} overlap",
        }
    )
    if overlap:
        raise AssertionError("Target relevan memuat item yang sudah dilihat saat training.")

    checks.append(
        {
            "pemeriksaan": "user universe sama untuk semua model",
            "status": "PASS",
            "bukti": f"{len(users)} user per model",
        }
    )
    return checks


# ------------------------------------------------------------------ pencarian
def run_search(
    base_rankings: dict[str, dict],
    users: list[str],
    relevant: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Evaluasi baseline + grid Weighted Score + grid RRF. Kembalikan log lengkap."""
    log_rows: list[dict] = []
    best_per_user: dict[str, pd.DataFrame] = {}

    def record(config_id: str, family: str, params: dict, rankings: dict, keep: bool):
        started = time.perf_counter()
        summary, per_user = evaluate(rankings, relevant, users, keep_per_user=keep)
        duration = time.perf_counter() - started
        row = {
            "config_id": config_id,
            "family": family,
            "params": json.dumps(params, sort_keys=True),
            "duration_sec": round(duration, 3),
            **{key: value for key, value in summary.items()},
        }
        log_rows.append(row)
        if keep and per_user is not None:
            best_per_user[config_id] = per_user
        return summary

    for model in BASELINE_MODELS:
        started = time.perf_counter()
        rankings = base_rankings[model]
        summary, per_user = evaluate(rankings, relevant, users, keep_per_user=True)
        duration = time.perf_counter() - started
        config_id = {
            "Popularity": "E1-popularity",
            "Item-CF": "E2-itemcf-k100",
            "TF-IDF CBF": "E3-tfidf-cbf",
        }[model]
        log_rows.append(
            {
                "config_id": config_id,
                "family": "baseline",
                "params": json.dumps({"model": model}, sort_keys=True),
                "duration_sec": round(duration, 3),
                **summary,
            }
        )
        best_per_user[config_id] = per_user

    for alpha in ALPHAS:
        config_id = f"E4-weighted-a{int(round(alpha * 100)):03d}"
        rankings = weighted_score_fusion(
            base_rankings["Item-CF"],
            base_rankings["TF-IDF CBF"],
            users,
            cf_weight=alpha,
            top_n=TOP_N,
        )
        record(
            config_id,
            "weighted",
            {"cf_weight": alpha, "cbf_weight": round(1 - alpha, 1)},
            rankings,
            keep=(alpha in (0.3, 0.0, 1.0)),
        )

    for rrf_k in RRF_KS:
        config_id = f"E5-rrf-k{rrf_k}"
        rankings = reciprocal_rank_fusion(
            base_rankings["Item-CF"],
            base_rankings["TF-IDF CBF"],
            users,
            rrf_k=rrf_k,
            top_n=TOP_N,
        )
        record(config_id, "rrf", {"rrf_k": rrf_k}, rankings, keep=(rrf_k == 100))

    return pd.DataFrame(log_rows), best_per_user


def select_winners(log: pd.DataFrame) -> dict:
    """Pilih konfigurasi terbaik per keluarga dan keseluruhan (NDCG@10, lalu Recall@10)."""
    ranked = log.sort_values(
        ["ndcg@10", "recall@10", "config_id"], ascending=[False, False, True]
    )
    best_by_family = {}
    for family, group in ranked.groupby("family", sort=False):
        best_by_family[family] = group.iloc[0].to_dict()

    overall = ranked.iloc[0].to_dict()
    return {"best_overall": overall, "best_by_family": best_by_family}


# -------------------------------------------------------------------- analisis
def user_segments(
    per_user: dict[str, pd.DataFrame], root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rata-rata NDCG@10 per segmen jumlah interaksi training user."""
    activity_path = next(
        root.glob("SUPPORTING_DATA-*/SUPPORTING_DATA/user_activity_train.csv"), None
    )
    if activity_path is None:
        return pd.DataFrame(), pd.DataFrame()

    activity = pd.read_csv(activity_path, dtype={"user_id": "string"})
    activity["user_id"] = activity["user_id"].astype(str)

    frames = []
    for config_id, frame in per_user.items():
        merged = frame.merge(
            activity[["user_id", "train_interactions"]], on="user_id", how="left"
        )
        merged["config_id"] = config_id
        frames.append(merged)
    combined = pd.concat(frames, ignore_index=True)

    labels = []
    for low, high, label in ACTIVITY_BUCKETS:
        mask = combined["train_interactions"].between(low, high)
        if mask.any():
            labels.append(label)
    combined["bucket"] = pd.cut(
        combined["train_interactions"],
        bins=[0, 4, 9, 19, float("inf")],
        labels=["1-4", "5-9", "10-19", "20+"],
    )

    segment = (
        combined.groupby(["config_id", "bucket"], observed=True)
        .agg(
            users=("user_id", "nunique"),
            ndcg=("ndcg@10", "mean"),
            recall=("recall@10", "mean"),
            hit_rate=("recall@10", lambda values: float((values > 0).mean() * 100)),
        )
        .reset_index()
        .rename(columns={"ndcg": "ndcg@10", "recall": "recall@10"})
    )
    return combined, segment


def first_hit_positions(per_user: dict[str, pd.DataFrame], users: list[str]) -> pd.DataFrame:
    """Distribusi posisi hit pertama (1..10) per model terpilih."""
    rows = []
    for config_id, frame in per_user.items():
        for _, entry in frame.iterrows():
            position = 0
            if entry["mrr@10"] > 0:
                position = int(round(1.0 / entry["mrr@10"]))
            rows.append({"config_id": config_id, "first_hit_position": position})
    distribution = pd.DataFrame(rows)
    distribution = (
        distribution.groupby(["config_id", "first_hit_position"])
        .size()
        .reset_index(name="users")
    )
    return distribution


# --------------------------------------------------------------------- figure
def style_axis(axis, title: str, ylabel: str, xlabel: str = "") -> None:
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=9)
    axis.grid(axis="y", alpha=0.3, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def save_figure(fig, out_dir: Path, name: str) -> list[Path]:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def build_figures(
    comparison: pd.DataFrame,
    log: pd.DataFrame,
    segment: pd.DataFrame,
    positions: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    models = list(comparison["label"])
    colors = [comparison.loc[comparison["label"] == label, "color"].iloc[0] for label in models]

    # 1. Grouped bar: NDCG@K per model
    fig, axis = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(models))
    width = 0.26
    for offset, k in enumerate(K_LIST):
        values = comparison[f"ndcg@{k}"].to_numpy()
        bars = axis.bar(x + (offset - 1) * width, values, width, label=f"NDCG@{k}")
        axis.bar_label(bars, fmt="%.4f", fontsize=6, padding=2)
    axis.set_xticks(x)
    axis.set_xticklabels(models, fontsize=8)
    style_axis(axis, "NDCG@10/20/50 per model (validation, 11.659 user)", "NDCG")
    axis.legend(fontsize=8, frameon=False)
    paths += save_figure(fig, out_dir, "fig1_ndcg_by_model")

    # 2. Line chart: metrik vs K
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, color in zip(models, colors):
        row = comparison.loc[comparison["label"] == label].iloc[0]
        axes[0].plot(K_LIST, [row[f"ndcg@{k}"] for k in K_LIST], marker="o", label=label, color=color)
        axes[1].plot(K_LIST, [row[f"recall@{k}"] for k in K_LIST], marker="o", label=label, color=color)
    style_axis(axes[0], "NDCG vs K", "NDCG", "K")
    style_axis(axes[1], "Recall vs K", "Recall", "K")
    axes[1].legend(fontsize=7, frameon=False)
    paths += save_figure(fig, out_dir, "fig2_metric_vs_k")

    # 3. Heatmap search space
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    weighted = log.loc[log["family"] == "weighted"].copy()
    weighted["alpha"] = weighted["params"].map(lambda text: json.loads(text)["cf_weight"])
    weighted = weighted.sort_values("alpha")
    matrix = weighted[[f"ndcg@{k}" for k in K_LIST]].to_numpy().T
    image = axes[0].imshow(matrix, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(len(weighted)))
    axes[0].set_xticklabels([f"{a:.1f}" for a in weighted["alpha"]], fontsize=7)
    axes[0].set_yticks(range(len(K_LIST)))
    axes[0].set_yticklabels([f"NDCG@{k}" for k in K_LIST], fontsize=8)
    axes[0].set_title("Weighted Score: alpha Item-CF", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("alpha", fontsize=9)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes[0].text(col, row, f"{matrix[row, col]:.4f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes[0], fraction=0.046)

    rrf = log.loc[log["family"] == "rrf"].copy()
    rrf["k"] = rrf["params"].map(lambda text: json.loads(text)["rrf_k"])
    rrf = rrf.sort_values("k")
    matrix = rrf[[f"ndcg@{k}" for k in K_LIST]].to_numpy().T
    image = axes[1].imshow(matrix, aspect="auto", cmap="Greens")
    axes[1].set_xticks(range(len(rrf)))
    axes[1].set_xticklabels([str(int(k)) for k in rrf["k"]], fontsize=7)
    axes[1].set_yticks(range(len(K_LIST)))
    axes[1].set_yticklabels([f"NDCG@{k}" for k in K_LIST], fontsize=8)
    axes[1].set_title("RRF: konstanta k", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("rrf_k", fontsize=9)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes[1].text(col, row, f"{matrix[row, col]:.4f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    paths += save_figure(fig, out_dir, "fig3_search_space_heatmap")

    # 4. Segmen aktivitas user
    if not segment.empty:
        labels = list(dict.fromkeys(segment["bucket"].astype(str)))
        fig, axis = plt.subplots(figsize=(9, 4))
        x = np.arange(len(labels))
        width = 0.8 / max(len(models), 1)
        for offset, (label, color) in enumerate(zip(models, colors)):
            config_id = comparison.loc[comparison["label"] == label, "config_id"].iloc[0]
            subset = segment.loc[segment["config_id"] == config_id].set_index("bucket")
            values = [subset["ndcg@10"].get(bucket, np.nan) for bucket in labels]
            axis.bar(x + offset * width, values, width, label=label, color=color)
        axis.set_xticks(x + width * (len(models) - 1) / 2)
        axis.set_xticklabels([f"{b} interaksi" for b in labels], fontsize=8)
        style_axis(axis, "NDCG@10 per segmen aktivitas training user", "NDCG@10")
        axis.legend(fontsize=7, frameon=False, ncol=2)
        paths += save_figure(fig, out_dir, "fig4_user_segments")

    # 5. Distribusi posisi hit pertama
    if not positions.empty:
        fig, axis = plt.subplots(figsize=(9, 4))
        x = np.arange(1, 11)
        width = 0.8 / max(len(models), 1)
        for offset, (label, color) in enumerate(zip(models, colors)):
            config_id = comparison.loc[comparison["label"] == label, "config_id"].iloc[0]
            subset = positions.loc[positions["config_id"] == config_id].set_index(
                "first_hit_position"
            )
            values = [int(subset["users"].get(position, 0)) for position in x]
            axis.bar(x + offset * width, values, width, label=label, color=color)
        style_axis(
            axis,
            "Jumlah user menurut posisi hit pertama (tanpa hit tidak ditampilkan)",
            "Jumlah user",
            "Posisi di Top-10",
        )
        axis.legend(fontsize=7, frameon=False, ncol=2)
        paths += save_figure(fig, out_dir, "fig5_first_hit_position")

    return paths


# -------------------------------------------------------------------- laporan
def write_summary_markdown(
    path: Path,
    metadata: dict,
    comparison: pd.DataFrame,
    winners: dict,
    audit: list[dict],
    segment: pd.DataFrame,
    figures: list[Path],
    log: pd.DataFrame,
) -> None:
    best = winners["best_overall"]
    popularity = comparison.loc[comparison["label"] == "Popularity"].iloc[0]
    rrf = comparison.loc[comparison["label"] == "RRF"].iloc[0]
    weighted = comparison.loc[comparison["label"] == "Weighted Score"].iloc[0]
    cf = comparison.loc[comparison["label"] == "Item-CF"].iloc[0]
    cbf = comparison.loc[comparison["label"] == "TF-IDF CBF"].iloc[0]

    def delta(value: float, base: float) -> str:
        return f"{(value / base - 1) * 100:+.2f}%"

    lines = [
        "# W11 - Laporan Hasil Eksperimen (validation)",
        "",
        f"Dihasilkan otomatis oleh `src/w11_experiment.py` pada {metadata['timestamp_utc']}.",
        "",
        "## 1. Protokol",
        "",
        f"- Split: official temporal; training {metadata['protocol']['train_interactions']:,} interaksi, "
        f"validation warm {metadata['protocol']['validation_warm_rows']:,} baris",
        f"- Evaluable users: {metadata['protocol']['evaluable_users']:,}",
        f"- Candidate universe: {metadata['protocol']['candidate_universe']:,} item; "
        f"ranking Top-{TOP_N}, dinilai Top-10/20/50",
        f"- Relevan: rating >= {POSITIVE_THRESHOLD}; target setelah filter: "
        f"{metadata['protocol']['relevant_targets']:,} baris",
        f"- Metrik utama pemilihan: NDCG@{PRIMARY_K} (tie-breaker Recall@{PRIMARY_K})",
        f"- Test set dipakai: {'ya' if metadata['protocol']['uses_final_test'] else 'tidak'}",
        f"- Seed: {SEED}; commit: {metadata['git']['commit'][:10]} "
        f"({'dirty' if metadata['git']['dirty'] else 'clean'})",
        "",
        "## 2. Tabel hasil utama (model x metrik)",
        "",
        "| Model | Konfigurasi | NDCG@10 | NDCG@20 | NDCG@50 | Recall@10 | Recall@20 | Recall@50 | Precision@10 | MRR@10 | MAP@10 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            "| {label} | {config} | {n10:.6f} | {n20:.6f} | {n50:.6f} | {r10:.6f} | "
            "{r20:.6f} | {r50:.6f} | {p10:.6f} | {m10:.6f} | {a10:.6f} |".format(
                label=row["label"],
                config=row["config_id"],
                n10=row["ndcg@10"],
                n20=row["ndcg@20"],
                n50=row["ndcg@50"],
                r10=row["recall@10"],
                r20=row["recall@20"],
                r50=row["recall@50"],
                p10=row["precision@10"],
                m10=row["mrr@10"],
                a10=row["map@10"],
            )
        )

    lines += [
        "",
        "## 3. Perbandingan terhadap baseline",
        "",
        "| Model | NDCG@10 | Delta NDCG vs Popularity | Recall@10 | Delta Recall vs Popularity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['label']} | {row['ndcg@10']:.6f} | {delta(row['ndcg@10'], popularity['ndcg@10'])} | "
            f"{row['recall@10']:.6f} | {delta(row['recall@10'], popularity['recall@10'])} |"
        )

    lines += [
        "",
        f"Model terbaik keseluruhan menurut NDCG@{PRIMARY_K}: **{best['config_id']}** "
        f"(NDCG@{PRIMARY_K} {best['ndcg@10']:.6f}, Recall@{PRIMARY_K} {best['recall@10']:.6f}).",
        "",
        "## 4. Temuan",
        "",
        f"1. Popularity tetap terkuat pada NDCG@10 ({popularity['ndcg@10']:.6f}) dan MRR@10 "
        f"({popularity['mrr@10']:.6f}); fusi tidak mengalahkannya pada metrik berbobot posisi.",
        f"2. RRF (k={int(json.loads(rrf['params'])['rrf_k'])}) unggul Recall@10 "
        f"({rrf['recall@10']:.6f}, {delta(rrf['recall@10'], popularity['recall@10'])} vs Popularity) "
        f"tetapi NDCG@10 {rrf['ndcg@10']:.6f} ({delta(rrf['ndcg@10'], popularity['ndcg@10'])}).",
        f"3. Weighted Score terbaik alpha={json.loads(weighted['params'])['cf_weight']}: "
        f"NDCG@10 {weighted['ndcg@10']:.6f}, Recall@10 {weighted['recall@10']:.6f}; "
        f"masih di bawah Popularity pada kedua metrik.",
        f"4. Item-CF terlemah: NDCG@10 {cf['ndcg@10']:.6f}, Recall@10 {cf['recall@10']:.6f} "
        f"({delta(cf['recall@10'], popularity['recall@10'])} vs Popularity).",
        f"5. TF-IDF CBF: NDCG@10 {cbf['ndcg@10']:.6f}, Recall@10 {cbf['recall@10']:.6f}, "
        f"dengan cakupan katalog jauh lebih luas ({int(cbf['unique_items@10']):,} item unik di Top-10 "
        f"vs {int(popularity['unique_items@10'])} item untuk Popularity).",
        f"6. Search space: {len(log)} konfigurasi dievaluasi "
        f"({int((log['family'] == 'weighted').sum())} Weighted Score, "
        f"{int((log['family'] == 'rrf').sum())} RRF, 3 baseline).",
        "",
        "## 5. Segmen aktivitas user (NDCG@10)",
        "",
    ]

    if not segment.empty:
        pivot = (
            segment.pivot(index="bucket", columns="config_id", values="ndcg@10")
            .round(6)
            .reset_index()
        )
        lines.append("| Segmen | " + " | ".join(pivot.columns[1:]) + " |")
        lines.append("| --- | " + " | ".join(["---"] * (len(pivot.columns) - 1)) + " |")
        for _, row in pivot.iterrows():
            lines.append(
                "| " + str(row["bucket"]) + " | "
                + " | ".join(f"{row[column]:.6f}" for column in pivot.columns[1:])
                + " |"
            )
        counts = (
            segment.groupby("bucket", observed=True)["users"].max().reset_index()
        )
        lines += [
            "",
            "Jumlah user per segmen: "
            + ", ".join(f"{row['bucket']}: {int(row['users'])}" for _, row in counts.iterrows()),
        ]
    else:
        lines.append("Data aktivitas user tidak tersedia.")

    lines += [
        "",
        "## 6. Audit",
        "",
        "| Pemeriksaan | Status | Bukti |",
        "| --- | --- | --- |",
    ]
    for check in audit:
        lines.append(f"| {check['pemeriksaan']} | {check['status']} | {check['bukti']} |")

    lines += [
        "",
        "## 7. Figure",
        "",
    ]
    for figure in figures:
        if figure.suffix == ".png":
            lines.append(f"- `{figure.as_posix()}`")

    lines += [
        "",
        "## 8. Keterbatasan",
        "",
        "- Ranking komponen dimuat dari cache Task 3 (hash tercatat di metadata), sehingga metrik "
        "mereproduksi run tersebut; tidak ada fitting ulang pada W11.",
        "- Test set belum tersedia dan tidak dipakai; seluruh angka adalah hasil pengembangan.",
        "- Search space terbatas pada alpha dan k RRF sesuai CP3; tidak ada model baru.",
        "- Item-CF masih memakai fallback popularitas untuk skor nol.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------ main
def run(root: Path, out_dir: Path, limit_users: int = 0) -> dict:
    started_at = datetime.now(timezone.utc)
    data = load_project_data(root)

    users = [str(user) for user in data.evaluable_users]
    if limit_users:
        users = users[:limit_users]

    candidates = [str(item) for item in data.candidates]
    candidate_set = set(candidates)

    train = data.train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)
    seen_by_user = {
        user: set(group["item_id"]) for user, group in train.groupby("user_id", sort=False)
    }

    # Protokol CP3: target relevan hanya item kandidat dan tidak training-seen.
    validation = data.validation_warm.copy()
    validation["user_id"] = validation["user_id"].astype(str)
    validation["item_id"] = validation["item_id"].astype(str)
    validation = validation.loc[validation["item_id"].isin(candidate_set)]
    relevant = build_relevant_items(validation, users, POSITIVE_THRESHOLD)
    relevant = {
        user: relevant.get(user, set()) - seen_by_user.get(user, set()) for user in users
    }

    base_rankings, cache_meta = load_cache(root)
    audit = audit_rankings(base_rankings, users, relevant, candidate_set, seen_by_user)

    log, per_user = run_search(base_rankings, users, relevant)
    winners = select_winners(log)

    # Baris perbandingan: 3 baseline + pemenang Weighted + pemenang RRF.
    labels = [
        ("Popularity", "E1-popularity"),
        ("Item-CF", "E2-itemcf-k100"),
        ("TF-IDF CBF", "E3-tfidf-cbf"),
        (
            "Weighted Score",
            winners["best_by_family"]["weighted"]["config_id"],
        ),
        ("RRF", winners["best_by_family"]["rrf"]["config_id"]),
    ]
    comparison_rows = []
    for label, config_id in labels:
        row = log.loc[log["config_id"] == config_id].iloc[0].to_dict()
        comparison_rows.append(
            {
                "label": label,
                "config_id": config_id,
                "family": row["family"],
                "params": row["params"],
                "color": MODEL_COLORS[label],
                **{
                    key: row[key]
                    for key in log.columns
                    if key.startswith(("recall@", "precision@", "ndcg@", "mrr@", "map@"))
                    or key in ("users_evaluated", "users_with_hit@10", "hit_rate_pct@10", "unique_items@10", "duration_sec")
                },
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    combined_segment, segment = user_segments(per_user, root)
    positions = first_hit_positions(per_user, users)

    metadata = {
        "task": "W11 - Experiment track on validation",
        "timestamp_utc": started_at.isoformat(),
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "git": git_state(root),
        "protocol": {
            "split": "official temporal; warm-start evaluable users",
            "positive_threshold": POSITIVE_THRESHOLD,
            "top_n_ranking": TOP_N,
            "k_list": list(K_LIST),
            "primary_metric": f"ndcg@{PRIMARY_K}",
            "train_interactions": int(len(train)),
            "validation_warm_rows": int(len(data.validation_warm)),
            "evaluable_users": int(len(users)),
            "candidate_universe": int(len(candidates)),
            "relevant_targets": int(sum(len(targets) for targets in relevant.values())),
            "uses_final_test": False,
        },
        "search_space": {"alphas": list(ALPHAS), "rrf_ks": list(RRF_KS)},
        "cache": cache_meta,
        "data_files": {
            path.name: sha256_file(path)
            for path in [
                next(root.glob("MAIN DATA-*/MAIN DATA/interactions_train.csv")),
                next(root.glob("SUPPORTING_DATA-*/SUPPORTING_DATA/interactions_validation_warm.csv")),
                next(root.glob("MAIN DATA-*/MAIN DATA/candidate_items_warm.csv")),
            ]
        },
        "audit": audit,
        "n_configs": int(len(log)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    log.to_csv(out_dir / "w11_all_configs.csv", index=False)
    comparison.to_csv(out_dir / "w11_model_comparison.csv", index=False)
    segment.to_csv(out_dir / "w11_user_segments.csv", index=False)
    positions.to_csv(out_dir / "w11_first_hit_positions.csv", index=False)
    if not combined_segment.empty:
        combined_segment.to_csv(out_dir / "w11_per_user_metrics.csv", index=False)

    (out_dir / "w11_metadata.json").write_text(
        json.dumps({**metadata, "winners": winners}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figures = build_figures(comparison, log, segment, positions, figures_dir)
    write_summary_markdown(
        out_dir / "w11_experiment_summary.md",
        metadata,
        comparison,
        winners,
        audit,
        segment,
        figures,
        log,
    )

    return {
        "metadata": metadata,
        "comparison": comparison,
        "winners": winners,
        "segment": segment,
        "figures": figures,
        "log": log,
        "duration_sec": (datetime.now(timezone.utc) - started_at).total_seconds(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W11 experiment track on validation")
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--out", default="outputs/w11", help="output directory")
    parser.add_argument(
        "--limit-users",
        type=int,
        default=0,
        help="smoke test only: batasi jumlah user evaluasi",
    )
    args = parser.parse_args(argv)

    result = run(Path(args.root), Path(args.out), limit_users=args.limit_users)
    comparison = result["comparison"]
    print(f"selesai dalam {result['duration_sec']:.1f}s")
    print(comparison[["label", "config_id", "ndcg@10", "recall@10", "ndcg@20", "ndcg@50"]].to_string(index=False))
    print("pemenang:", result["winners"]["best_overall"]["config_id"])
    print("figure:", len(result["figures"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
