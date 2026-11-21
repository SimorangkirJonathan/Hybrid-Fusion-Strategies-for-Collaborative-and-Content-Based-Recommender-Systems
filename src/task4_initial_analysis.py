"""CP3 3.4 - pemeriksaan dan analisis hasil awal.

Modul ini hanya membaca artefak yang sudah dihasilkan Task 2 dan Task 3.
Tidak melatih model baru, tidak menghitung ulang metrik, dan tidak menyentuh test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRICS = ("recall@10", "ndcg@10")
OUTPUT_DIR = Path("outputs") / "task4"
TOLERANCE = 1e-9


def load_artifacts(root: str | Path = ".") -> dict:
    """Baca semua artefak pembanding yang sudah ada."""
    root = Path(root)
    task2 = root / "outputs" / "task2"
    task3 = root / "outputs" / "task3"
    return {
        "per_user": pd.read_csv(task2 / "baseline_metrics_per_user.csv"),
        "top10": pd.read_csv(task2 / "baseline_top10.csv"),
        "baseline_summary": pd.read_csv(task2 / "baseline_comparison.csv"),
        "protocol": json.loads(
            (task2 / "task2_protocol_and_results.json").read_text(encoding="utf-8")
        ),
        "model_comparison": pd.read_csv(task3 / "task3_model_comparison.csv"),
        "task3_summary": json.loads(
            (task3 / "task3_summary.json").read_text(encoding="utf-8")
        ),
        "weighted": pd.read_csv(task3 / "task3_weighted_score_results.csv"),
        "rrf": pd.read_csv(task3 / "task3_rrf_results.csv"),
    }


def audit_artifacts(artifacts: dict) -> dict:
    """Pastikan artefak yang dibandingkan memang memakai protokol yang sama."""
    per_user = artifacts["per_user"]
    summary3 = artifacts["task3_summary"]
    protocol = artifacts["protocol"]

    models = sorted(per_user["model"].unique())
    assert len(models) == 3, f"Harus ada 3 baseline, ditemukan {models}."

    counts = per_user.groupby("model")["user_id"].nunique()
    assert counts.nunique() == 1, f"Jumlah user antar baseline berbeda: {counts.to_dict()}."
    users = int(counts.iloc[0])
    assert users == protocol["evaluable_users"], "Jumlah user beda dengan protokol Task 2."
    assert users == summary3["evaluable_users"], "Jumlah user beda dengan ringkasan Task 3."

    assert not per_user[list(METRICS)].isna().any().any(), "Ada metrik kosong."
    assert per_user[list(METRICS)].ge(0).all().all(), "Ada metrik negatif."
    assert per_user[list(METRICS)].le(1 + 1e-9).all().all(), "Ada metrik di atas 1."

    popularity_sources = [
        float(per_user.loc[per_user["model"] == "Popularity", "recall@10"].mean()),
        float(summary3["popularity_baseline"]["recall@10"]),
        float(protocol["results"][0]["recall@10"]),
    ]
    spread = max(popularity_sources) - min(popularity_sources)
    assert spread < TOLERANCE, f"Popularity tidak konsisten antar artefak: {popularity_sources}."

    baseline_labels = {"Popularity", "Item-CF", "TF-IDF CBF"}
    present = set(artifacts["model_comparison"]["Model"].astype(str))
    assert baseline_labels.issubset(present), f"Baseline hilang di Task 3: {baseline_labels - present}"

    return {
        "users": users,
        "baseline_models": models,
        "popularity_recall_sources": popularity_sources,
        "protocol_k": protocol.get("k"),
        "protocol_positive_threshold": protocol.get("positive_threshold"),
        "uses_test_data": bool(protocol.get("uses_final_test")) or bool(summary3.get("uses_final_test")),
    }


def summarize_models(artifacts: dict) -> pd.DataFrame:
    """Tabel perbandingan baseline (per user) plus model fusi (ringkasan saja)."""
    per_user = artifacts["per_user"]
    top10 = artifacts["top10"]
    comparison = artifacts["model_comparison"].set_index("Model")
    users = int(per_user["user_id"].nunique())

    rows = []
    for model, group in per_user.groupby("model", sort=False):
        recs = top10.loc[top10["model"] == model]
        rows.append(
            {
                "model": model,
                "sumber_per_user": True,
                "users": int(group["user_id"].nunique()),
                "recall@10": float(group["recall@10"].mean()),
                "ndcg@10": float(group["ndcg@10"].mean()),
                "users_with_hit": int((group["recall@10"] > 0).sum()),
                "hit_rate_pct": float((group["recall@10"] > 0).mean() * 100),
                "mean_targets": float(group["n_relevant"].mean()),
                "unique_items_top10": int(recs["item_id"].nunique()),
                "unique_items_rank1": int(recs.loc[recs["rank"] == 1, "item_id"].nunique()),
            }
        )

    known = set(per_user["model"].unique())
    for label, row in comparison.iterrows():
        if str(label) in known:
            continue
        rows.append(
            {
                "model": str(label),
                "sumber_per_user": False,
                "users": users,
                "recall@10": float(row["recall@10"]),
                "ndcg@10": float(row["ndcg@10"]),
                "users_with_hit": None,
                "hit_rate_pct": None,
                "mean_targets": None,
                "unique_items_top10": None,
                "unique_items_rank1": None,
            }
        )

    table = pd.DataFrame(rows)
    popularity_recall = float(
        table.loc[table["model"] == "Popularity", "recall@10"].iloc[0]
    )
    table["recall_vs_popularity_pct"] = (table["recall@10"] / popularity_recall - 1) * 100
    return table.sort_values("recall@10", ascending=False).reset_index(drop=True)


def identify_extremes(table: pd.DataFrame) -> dict:
    """Baseline terkuat/terlemah menurut masing-masing metrik."""
    baselines = table.loc[table["sumber_per_user"]].copy()
    best_recall = baselines.loc[baselines["recall@10"].idxmax()]
    best_ndcg = baselines.loc[baselines["ndcg@10"].idxmax()]
    worst_recall = baselines.loc[baselines["recall@10"].idxmin()]
    worst_ndcg = baselines.loc[baselines["ndcg@10"].idxmin()]
    return {
        "terkuat_recall": str(best_recall["model"]),
        "terkuat_ndcg": str(best_ndcg["model"]),
        "terlemah_recall": str(worst_recall["model"]),
        "terlemah_ndcg": str(worst_ndcg["model"]),
        "rasio_terlemah_terkuat_recall": float(
            worst_recall["recall@10"] / best_recall["recall@10"]
        ),
    }


def user_overlap(per_user: pd.DataFrame) -> pd.DataFrame:
    """Berapa user yang mendapat minimal satu hit, per kombinasi model."""
    pivot = per_user.pivot(index="user_id", columns="model", values="recall@10").fillna(0.0)
    hit = pivot > 0

    def label(row: pd.Series) -> str:
        names = [model for model in hit.columns if bool(row[model])]
        return " + ".join(names) if names else "tidak ada hit"

    combos = hit.apply(label, axis=1).value_counts()
    frame = (
        combos.rename_axis("kombinasi_model_dengan_hit")
        .reset_index(name="n_users")
        .sort_values(["n_users", "kombinasi_model_dengan_hit"], ascending=[False, True])
    )
    return frame.reset_index(drop=True)


def pairwise_wins(per_user: pd.DataFrame) -> pd.DataFrame:
    """Per bandingan berpasangan: berapa user model A lebih baik dari model B."""
    pivot = per_user.pivot(index="user_id", columns="model", values="recall@10").fillna(0.0)
    models = list(pivot.columns)
    rows = []
    for index, model_a in enumerate(models):
        for model_b in models[index + 1:]:
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "a_lebih_baik": int((pivot[model_a] > pivot[model_b]).sum()),
                    "b_lebih_baik": int((pivot[model_b] > pivot[model_a]).sum()),
                    "seri": int((pivot[model_a] == pivot[model_b]).sum()),
                }
            )
    return pd.DataFrame(rows)


def detect_anomalies(artifacts: dict, table: pd.DataFrame) -> list[dict]:
    """Kumpulkan temuan yang perlu dijelaskan di laporan."""
    anomalies: list[dict] = []
    base = artifacts["baseline_summary"].set_index("model")
    comparison = artifacts["model_comparison"].set_index("Model")

    for model in base.index:
        if model not in comparison.index:
            continue
        differences = []
        for metric in METRICS:
            task2_value = float(base.loc[model, metric])
            task3_value = float(comparison.loc[model, metric])
            if abs(task2_value - task3_value) > TOLERANCE:
                differences.append(
                    f"{metric} = {task2_value:.7f} di baseline_comparison.csv vs "
                    f"{task3_value:.7f} di task3_model_comparison.csv "
                    f"(selisih {abs(task2_value - task3_value):.2e}, "
                    f"{abs(task3_value / task2_value - 1) * 100:.2f}% relatif)"
                )
        if differences:
            anomalies.append(
                {
                    "id": "nilai_berbeda_" + model.lower().replace("-", "_").replace(" ", "_"),
                    "judul": f"Nilai {model} berbeda antara artefak Task 2 dan Task 3",
                    "detail": "; ".join(differences)
                    + ". Kemungkinan cache rekomendasi dipakai ulang setelah kode berubah; "
                    "hapus outputs/task3/cache lalu jalankan ulang sebelum angka dikutip.",
                    "dampak": "angka CF/CBF belum bisa dikutip sebagai satu nilai final",
                }
            )

    popularity = comparison.loc["Popularity"]
    rrf_label = next((label for label in comparison.index if str(label).startswith("RRF")), None)
    weighted_label = next(
        (label for label in comparison.index if str(label).startswith("Weighted Score")), None
    )
    if rrf_label is not None:
        rrf = comparison.loc[rrf_label]
        if rrf["recall@10"] > popularity["recall@10"] and rrf["ndcg@10"] < popularity["ndcg@10"]:
            anomalies.append(
                {
                    "id": "tradeoff_rrf_recall_vs_ndcg",
                    "judul": "RRF unggul Recall@10 tetapi kalah NDCG@10 dari Popularity",
                    "detail": (
                        f"Recall@10 {rrf['recall@10']:.6f} vs {popularity['recall@10']:.6f} "
                        f"(+{(rrf['recall@10'] / popularity['recall@10'] - 1) * 100:.2f}%), "
                        f"NDCG@10 {rrf['ndcg@10']:.6f} vs {popularity['ndcg@10']:.6f} "
                        f"({(rrf['ndcg@10'] / popularity['ndcg@10'] - 1) * 100:.2f}%). "
                        "Fusi menemukan lebih banyak item relevan, tetapi menempatkannya lebih rendah."
                    ),
                    "dampak": "pemilihan model tidak bisa hanya memakai satu metrik",
                }
            )
    if weighted_label is not None:
        weighted = comparison.loc[weighted_label]
        if weighted["recall@10"] < popularity["recall@10"]:
            anomalies.append(
                {
                    "id": "weighted_score_di_bawah_popularity",
                    "judul": "Weighted Score terbaik masih di bawah Popularity",
                    "detail": (
                        f"{weighted_label}: Recall@10 {weighted['recall@10']:.6f} vs "
                        f"{popularity['recall@10']:.6f}. Bobot 30% untuk Item-CF menarik "
                        "kualitas ranking turun karena komponen CF adalah baseline terlemah."
                    ),
                    "dampak": "bobot optimal cenderung menekan komponen CF, bukan memanfaatkannya",
                }
            )

    popularity_row = table.loc[table["model"] == "Popularity"].iloc[0]
    cf_row = table.loc[table["model"] == "Item-CF"].iloc[0]
    if popularity_row["unique_items_top10"] and cf_row["unique_items_top10"]:
        if popularity_row["unique_items_top10"] * 100 < cf_row["unique_items_top10"]:
            anomalies.append(
                {
                    "id": "popularity_tanpa_personalisasi",
                    "judul": "Popularity hanya memakai belasan item untuk seluruh pengguna",
                    "detail": (
                        f"Seluruh {int(popularity_row['users']):,} pengguna hanya menerima "
                        f"{int(popularity_row['unique_items_top10'])} item unik di Top-10, dan "
                        f"hanya {int(popularity_row['unique_items_rank1'])} item berbeda di peringkat 1. "
                        f"Item-CF menyentuh {int(cf_row['unique_items_top10']):,} item, "
                        f"TF-IDF CBF {int(table.loc[table['model'] == 'TF-IDF CBF', 'unique_items_top10'].iloc[0]):,} item."
                    ),
                    "dampak": "keunggulan recall Popularity dicapai dengan portfolio hampir seragam",
                }
            )

    single_target_share = float((artifacts["per_user"]["n_relevant"] == 1).mean() * 100)
    if single_target_share > 50:
        anomalies.append(
            {
                "id": "mayoritas_user_target_tunggal",
                "judul": "Mayoritas pengguna hanya punya satu item relevan",
                "detail": (
                    f"{single_target_share:.2f}% pengguna memiliki tepat satu target relevan, "
                    f"rata-rata {artifacts['per_user']['n_relevant'].mean():.2f} target per pengguna. "
                    "Untuk kelompok ini Recall@10 bersifat biner dan NDCG@10 hanya ditentukan "
                    "posisi satu item."
                ),
                "dampak": "perbedaan model kecil sulit dibedakan secara statistik",
            }
        )
    return anomalies


def build_insights(table: pd.DataFrame, overlap: pd.DataFrame, anomalies: list[dict]) -> list[str]:
    """Insight yang bisa langsung dipakai untuk perbaikan tahap berikutnya."""
    popularity = table.loc[table["model"] == "Popularity"].iloc[0]
    cf = table.loc[table["model"] == "Item-CF"].iloc[0]
    cbf = table.loc[table["model"] == "TF-IDF CBF"].iloc[0]

    def count(label: str) -> int:
        row = overlap.loc[overlap["kombinasi_model_dengan_hit"] == label, "n_users"]
        return int(row.iloc[0]) if len(row) else 0

    all_three = count("Item-CF + Popularity + TF-IDF CBF")
    popularity_only = count("Popularity")
    cbf_only = count("TF-IDF CBF")
    cf_only = count("Item-CF")

    insights = [
        (
            f"Item-CF adalah baseline terlemah: Recall@10 {cf['recall@10']:.6f}, sekitar "
            f"{(1 - cf['recall@10'] / popularity['recall@10']) * 100:.0f}% di bawah Popularity, "
            f"dan hanya {int(cf['users_with_hit'])} pengguna mendapat satu pun item relevan. "
            "Fallback popularitas di recommend_from_model membuat Top-10 Item-CF sebagian besar "
            "bukan hasil kemiripan item, jadi kontribusi CF murni perlu diukur terpisah."
        ),
        (
            f"Ketiga baseline hampir tidak beririsan: hanya {all_three} pengguna mendapat hit di "
            f"ketiganya, sedangkan {popularity_only} pengguna hanya tertolong Popularity, "
            f"{cbf_only} hanya TF-IDF CBF, dan {cf_only} hanya Item-CF. Ini menjelaskan mengapa "
            "fusi berbasis ranking bisa menaikkan Recall di atas semua komponennya."
        ),
        (
            "Prioritas perbaikan berikutnya adalah memperbaiki skor Item-CF (normalisasi jumlah "
            "tetangga, ambang co-occurrence minimum, dan memisahkan fallback popularitas sebagai "
            "varian tersendiri) sebelum menambah model baru seperti MF."
        ),
        (
            "Popularity sebaiknya ikut dijadikan komponen fusi, bukan hanya pembanding: karena "
            "ia menang pada pengguna yang tidak tertolong CF/CBF, daftar ketiga berpeluang "
            "memperbaiki NDCG fusi yang saat ini justru lebih rendah dari Popularity."
        ),
        (
            f"Pelaporan perlu ditambah hit-rate@10 dan catalog coverage. Popularity hanya "
            f"menyentuh {int(popularity['unique_items_top10'])} item unik, sementara "
            f"{float((table.loc[table['model'] == 'Popularity', 'hit_rate_pct'].iloc[0])):.2f}% "
            "pengguna mendapat hit; angka recall saja menyembunyikan perbedaan cakupan ini."
        ),
    ]
    if any(item["id"].startswith("nilai_berbeda") for item in anomalies):
        insights.append(
            "Sebelum angka CF/CBF dikutip di laporan, hapus outputs/task3/cache lalu jalankan "
            "ulang Task_3_Evaluasi_Eksperimen_Hybrid_FAST supaya nilainya sama dengan "
            "baseline_comparison.csv."
        )
    return insights


def _write_markdown(
    path: Path,
    audit: dict,
    table: pd.DataFrame,
    extremes: dict,
    overlap: pd.DataFrame,
    wins: pd.DataFrame,
    anomalies: list[dict],
    insights: list[str],
) -> None:
    def fmt(value, digits: int = 6) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}"

    def fmt_int(value) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{int(value)}"

    lines = [
        "# CP3 3.4 - Pemeriksaan dan Analisis Hasil Awal",
        "",
        "Dihasilkan otomatis dari artefak Task 2 dan Task 3 oleh `src/task4_initial_analysis.py`.",
        "",
        f"- Pengguna evaluasi: {audit['users']:,}",
        f"- Metrik: Recall@{audit['protocol_k']} dan NDCG@{audit['protocol_k']} (macro per pengguna)",
        f"- Ambang relevan: rating >= {audit['protocol_positive_threshold']}",
        f"- Data test dipakai: {'ya' if audit['uses_test_data'] else 'tidak'}",
        "",
        "## 1. Perbandingan performa",
        "",
        "| Model | Recall@10 | NDCG@10 | Selisih recall vs Popularity | Pengguna dengan hit | Item unik di Top-10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {model} | {recall} | {ndcg} | {delta} | {hits} | {items} |".format(
                model=row["model"],
                recall=fmt(row["recall@10"]),
                ndcg=fmt(row["ndcg@10"]),
                delta=f"{row['recall_vs_popularity_pct']:+.2f}%",
                hits=fmt_int(row["users_with_hit"]),
                items=fmt_int(row["unique_items_top10"]),
            )
        )

    lines += [
        "",
        "## 2. Terkuat dan terlemah",
        "",
        f"- Terkuat Recall@10: **{extremes['terkuat_recall']}**",
        f"- Terkuat NDCG@10: **{extremes['terkuat_ndcg']}**",
        f"- Terlemah Recall@10: **{extremes['terlemah_recall']}** "
        f"(rasio {extremes['rasio_terlemah_terkuat_recall']:.3f} dari yang terkuat)",
        f"- Terlemah NDCG@10: **{extremes['terlemah_ndcg']}**",
        "",
        "## 3. Temuan dan anomali",
        "",
    ]
    if anomalies:
        for index, item in enumerate(anomalies, start=1):
            lines += [
                f"{index}. **{item['judul']}**",
                f"   - {item['detail']}",
                f"   - Dampak: {item['dampak']}",
                "",
            ]
    else:
        lines += ["Tidak ada anomali terdeteksi.", ""]

    lines += [
        "### Irisan pengguna yang mendapat minimal satu hit",
        "",
        "| Kombinasi model | Jumlah pengguna |",
        "| --- | --- |",
    ]
    for _, row in overlap.iterrows():
        lines.append(f"| {row['kombinasi_model_dengan_hit']} | {int(row['n_users'])} |")

    lines += ["", "### Perbandingan berpasangan (Recall@10)", "", "| A | B | A lebih baik | B lebih baik | Seri |", "| --- | --- | --- | --- | --- |"]
    for _, row in wins.iterrows():
        lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['a_lebih_baik']} | "
            f"{row['b_lebih_baik']} | {row['seri']} |"
        )

    lines += ["", "## 4. Insight untuk tahap berikutnya", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(insights, start=1)]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_analysis(
    root: str | Path,
    artifacts: dict,
    audit: dict,
    table: pd.DataFrame,
    extremes: dict,
    overlap: pd.DataFrame,
    wins: pd.DataFrame,
    anomalies: list[dict],
    insights: list[str],
) -> list[Path]:
    """Simpan tabel, ringkasan JSON, dan laporan markdown."""
    root = Path(root)
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "task4_model_comparison.csv"
    overlap_path = output_dir / "task4_user_overlap.csv"
    wins_path = output_dir / "task4_pairwise_wins.csv"
    summary_path = output_dir / "task4_initial_analysis.json"
    report_path = output_dir / "task4_analysis_summary.md"

    table.to_csv(comparison_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    wins.to_csv(wins_path, index=False)

    payload = {
        "task": "CP3 3.4 - Pemeriksaan dan Analisis Hasil Awal",
        "dataset": "Amazon Reviews 2023 - Musical Instruments",
        "sources": [
            "outputs/task2/baseline_comparison.csv",
            "outputs/task2/baseline_metrics_per_user.csv",
            "outputs/task2/baseline_top10.csv",
            "outputs/task3/task3_model_comparison.csv",
            "outputs/task3/task3_summary.json",
            "outputs/task3/task3_weighted_score_results.csv",
            "outputs/task3/task3_rrf_results.csv",
        ],
        "audit": audit,
        "comparison": table.to_dict(orient="records"),
        "extremes": extremes,
        "user_overlap": overlap.to_dict(orient="records"),
        "pairwise_wins": wins.to_dict(orient="records"),
        "anomalies": anomalies,
        "insights": insights,
        "uses_test_data": audit["uses_test_data"],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report_path, audit, table, extremes, overlap, wins, anomalies, insights)
    return [comparison_path, overlap_path, wins_path, summary_path, report_path]


def run(root: str | Path = ".") -> dict:
    """Jalankan seluruh analisis 3.4 dan kembalikan hasilnya."""
    root = Path(root)
    artifacts = load_artifacts(root)
    audit = audit_artifacts(artifacts)
    table = summarize_models(artifacts)
    extremes = identify_extremes(table)
    overlap = user_overlap(artifacts["per_user"])
    wins = pairwise_wins(artifacts["per_user"])
    anomalies = detect_anomalies(artifacts, table)
    insights = build_insights(table, overlap, anomalies)
    paths = export_analysis(
        root, artifacts, audit, table, extremes, overlap, wins, anomalies, insights
    )
    return {
        "audit": audit,
        "table": table,
        "extremes": extremes,
        "overlap": overlap,
        "wins": wins,
        "anomalies": anomalies,
        "insights": insights,
        "paths": paths,
    }


if __name__ == "__main__":
    result = run(".")
    print("3.4 selesai. Berkas keluaran:")
    for path in result["paths"]:
        print("-", path.as_posix())
    print()
    print(result["table"].to_string(index=False))
    print()
    for item in result["anomalies"]:
        print(f"[ANOMALI] {item['judul']}")
    print()
    for line in result["insights"]:
        print(f"[INSIGHT] {line}")
