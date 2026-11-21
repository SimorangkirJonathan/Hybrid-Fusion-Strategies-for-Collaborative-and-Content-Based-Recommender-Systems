"""Membangun laporan hasil eksperimen W11 (PDF + markdown analisis RQ).

Membaca artefak `outputs/w11/` yang dihasilkan `src/w11_experiment.py`,
lalu menyusun:
- output/pdf/W11_Laporan_Hasil_Eksperimen.pdf
- outputs/w11/w11_rq_analysis.md  (teks siap tempel ke Overleaf)

Jalankan: python -m src.w11_build_report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(".")
DATA_DIR = ROOT / "outputs" / "w11"
FIG_DIR = DATA_DIR / "figures"
PDF_PATH = ROOT / "output" / "pdf" / "W11_Laporan_Hasil_Eksperimen.pdf"
MD_PATH = DATA_DIR / "w11_rq_analysis.md"

NAVY = colors.HexColor("#1b265e")
BLUE = colors.HexColor("#27459b")
LIGHT = colors.HexColor("#f2f4fa")
GRAY = colors.HexColor("#595959")

MODEL_ORDER = ["Popularity", "Item-CF", "TF-IDF CBF", "Weighted Score", "RRF"]


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    comparison = pd.read_csv(DATA_DIR / "w11_model_comparison.csv")
    configs = pd.read_csv(DATA_DIR / "w11_all_configs.csv")
    segments = pd.read_csv(DATA_DIR / "w11_user_segments.csv")
    metadata = json.loads((DATA_DIR / "w11_metadata.json").read_text(encoding="utf-8"))
    comparison["order"] = comparison["label"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    comparison = comparison.sort_values("order").drop(columns="order").reset_index(drop=True)
    return comparison, configs, segments, metadata


def pct(value: float, base: float) -> str:
    return f"{(value / base - 1) * 100:+.2f}%"


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=19,
            leading=23, textColor=NAVY, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
            leading=13, textColor=GRAY, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.5,
            leading=15, textColor=BLUE, spaceBefore=12, spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=NAVY, spaceBefore=8, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica", fontSize=9.2,
            leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica", fontSize=9.2,
            leading=13, leftIndent=12, bulletIndent=2, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.8,
            leading=10, textColor=GRAY, alignment=TA_LEFT, spaceBefore=2, spaceAfter=9,
        ),
    }


def header_table(comparison: pd.DataFrame) -> Table:
    header = ["Model", "NDCG@10", "NDCG@20", "NDCG@50", "Recall@10", "Precision@10", "MRR@10"]
    rows = [header]
    for _, row in comparison.iterrows():
        rows.append(
            [
                row["label"],
                fmt(row["ndcg@10"]),
                fmt(row["ndcg@20"]),
                fmt(row["ndcg@50"]),
                fmt(row["recall@10"]),
                fmt(row["precision@10"]),
                fmt(row["mrr@10"]),
            ]
        )
    table = Table(rows, colWidths=[3.9 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfe0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def baseline_table(comparison: pd.DataFrame) -> Table:
    popularity = comparison.loc[comparison["label"] == "Popularity"].iloc[0]
    header = ["Model", "NDCG@10", "Delta NDCG", "Recall@10", "Delta Recall", "Hit rate@10", "Item unik@10"]
    rows = [header]
    for _, row in comparison.iterrows():
        rows.append(
            [
                row["label"],
                fmt(row["ndcg@10"]),
                pct(row["ndcg@10"], popularity["ndcg@10"]),
                fmt(row["recall@10"]),
                pct(row["recall@10"], popularity["recall@10"]),
                f"{row['hit_rate_pct@10']:.2f}%",
                f"{int(row['unique_items@10']):,}".replace(",", "."),
            ]
        )
    table = Table(rows, colWidths=[3.4 * cm, 2.2 * cm, 2.3 * cm, 2.2 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfe0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def figure_flowable(name: str, width_cm: float = 15.8) -> Image:
    """Skala gambar sesuai rasio aslinya agar tidak melebihi frame A4."""
    width = width_cm * cm
    image = ImageReader(str(FIG_DIR / name))
    pixel_width, pixel_height = image.getSize()
    height = width * pixel_height / pixel_width
    return Image(str(FIG_DIR / name), width=width, height=height)


def analysis_paragraphs(comparison: pd.DataFrame, segments: pd.DataFrame) -> list[str]:
    row = {entry["label"]: entry for _, entry in comparison.iterrows()}
    pop, cf, cbf, ws, rrf = (
        row["Popularity"], row["Item-CF"], row["TF-IDF CBF"], row["Weighted Score"], row["RRF"]
    )
    seg = segments.set_index(["config_id", "bucket"])

    def segment(config_id: str, bucket: str, column: str = "ndcg@10") -> float:
        return float(seg.loc[(config_id, bucket), column])

    return [
        (
            "Temuan utama. Fusi hybrid berhasil menaikkan Recall@10 di atas seluruh baseline, "
            f"tetapi tidak pada NDCG@10. RRF (k=100) mencapai Recall@10 {fmt(rrf['recall@10'])} "
            f"({pct(rrf['recall@10'], pop['recall@10'])} dibanding Popularity), sementara NDCG@10 "
            f"{fmt(rrf['ndcg@10'])} masih {pct(rrf['ndcg@10'], pop['ndcg@10'])} di bawah Popularity. "
            "Artinya, fusi menemukan lebih banyak item relevan, tetapi menempatkannya pada posisi "
            "yang lebih rendah."
        ),
        (
            "a. Apakah track mengungguli baseline? Ya, tetapi hanya pada Recall@10. RRF unggul "
            f"{pct(rrf['recall@10'], pop['recall@10'])} atas Popularity, {pct(rrf['recall@10'], cbf['recall@10'])} "
            f"atas TF-IDF CBF, dan {pct(rrf['recall@10'], cf['recall@10'])} atas Item-CF. Pada NDCG@10 "
            f"RRF kalah {pct(rrf['ndcg@10'], pop['ndcg@10'])} dari Popularity, meskipun tetap "
            f"{pct(rrf['ndcg@10'], cbf['ndcg@10'])} di atas TF-IDF CBF dan {pct(rrf['ndcg@10'], cf['ndcg@10'])} "
            f"di atas Item-CF. Weighted Score terbaik (alpha=0,3) tidak mengungguli Popularity pada "
            f"kedua metrik, tetapi memberi perbaikan kecil atas komponen CBF-nya sendiri: "
            f"{pct(ws['ndcg@10'], cbf['ndcg@10'])} NDCG@10 dan {pct(ws['recall@10'], cbf['recall@10'])} Recall@10."
        ),
        (
            "b. Kapan model bekerja baik atau buruk? Popularity paling kuat pada user dengan sedikit "
            f"interaksi (segmen 1-4: NDCG@10 {fmt(segment('E1-popularity', '1-4'))}) dan melemah pada "
            f"user aktif (20+: {fmt(segment('E1-popularity', '20+'))}). RRF justru paling stabil dan "
            f"tertinggi pada user aktif (20+: {fmt(segment('E5-rrf-k100', '20+'))}; 10-19: "
            f"{fmt(segment('E5-rrf-k100', '10-19'))}), karena fusi memanfaatkan riwayat yang lebih panjang. "
            f"TF-IDF CBF dan Weighted Score turun tajam pada segmen 10-19 "
            f"({fmt(segment('E3-tfidf-cbf', '10-19'))} dan {fmt(segment('E4-weighted-a030', '10-19'))}) "
            "lalu pulih pada 20+. Item-CF terlemah di semua segmen, walau membaik seiring aktivitas "
            f"({fmt(segment('E2-itemcf-k100', '1-4'))} pada 1-4 menjadi {fmt(segment('E2-itemcf-k100', '20+'))} pada 20+)."
        ),
        (
            "c. Apakah hasil mendukung RQ? Sebagian. RQ menanyakan apakah fusi menghasilkan Recall@10 "
            "dan NDCG@10 yang lebih baik daripada Popularity, Item-CF, dan TF-IDF CBF. Untuk Recall@10, "
            "jawabannya ya dan konsisten dengan Tabel 2. Untuk NDCG@10, jawabannya belum: Popularity "
            f"tetap tertinggi ({fmt(pop['ndcg@10'])}), diikuti RRF ({fmt(rrf['ndcg@10'])}), Weighted Score "
            f"({fmt(ws['ndcg@10'])}), TF-IDF CBF ({fmt(cbf['ndcg@10'])}), dan Item-CF ({fmt(cf['ndcg@10'])}). "
            "Hipotesis bahwa fusi selalu lebih baik tidak didukung data ini."
        ),
        (
            "d. Temuan tak terduga. Popularity hanya memakai 18 item unik di seluruh Top-10, tetapi "
            "memenangi NDCG@10 dan MRR@10 "
            f"({fmt(pop['mrr@10'])} vs {fmt(rrf['mrr@10'])} untuk RRF), tanda bahwa sinyal personal "
            "pada data ini lemah dan posisi teratas didominasi item populer. Sebaliknya, keunggulan "
            f"Recall RRF datang dari cakupan katalog yang jauh lebih luas "
            f"({int(rrf['unique_items@10']):,} item unik vs {int(pop['unique_items@10'])}). "
            "Sensitivitas bobot juga tajam: NDCG@10 memuncak pada alpha=0,3 (0,007930) lalu merosot "
            "saat bobot Item-CF dinaikkan (alpha=0,5: 0,007411; alpha=0,7: 0,005039; alpha=1,0: 0,003338). "
            "Untuk RRF, NDCG@10 naik monoton terhadap k dan terbaik pada k=100 (0,009086)."
        ),
        (
            "e. Keterbatasan. (1) Ranking komponen dimuat dari cache Task 3 yang sudah diaudit, sehingga "
            "tidak ada fitting ulang pada W11; angka mereproduksi run tersebut secara persis. (2) Search "
            "space terbatas pada alpha dan k RRF sesuai CP3; Popularity belum dijadikan komponen fusi. "
            "(3) Item-CF masih memakai fallback popularitas untuk skor nol, sehingga kontribusi CF murni "
            "belum terisolasi. (4) Belum ada uji signifikansi, padahal 10.984 dari 11.659 user (94,2%) "
            "tidak memperoleh hit dan 54,5% hanya punya satu target. (5) Test set belum tersedia dan tidak "
            "dipakai; seluruh angka adalah hasil pengembangan di validation, dan evaluasi final dilakukan "
            "di W13."
        ),
    ]


def build_markdown(comparison: pd.DataFrame, paragraphs: list[str], metadata: dict) -> None:
    lines = [
        "# W11 - Analisis Hasil untuk Menjawab RQ",
        "",
        f"Semua angka dari `outputs/w11/` (validation warm, {metadata['protocol']['evaluable_users']:,} user, "
        f"test tidak dipakai). Metrik utama pemilihan: NDCG@10 sesuai CP3; K tambahan 20 dan 50.",
        "",
        "| Model | NDCG@10 | Recall@10 | NDCG@20 | NDCG@50 | Recall@50 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['ndcg@10'])} | {fmt(row['recall@10'])} | "
            f"{fmt(row['ndcg@20'])} | {fmt(row['ndcg@50'])} | {fmt(row['recall@50'])} |"
        )
    lines += ["", "## Analisis", ""]
    for index, paragraph in enumerate(paragraphs, start=1):
        lines += [paragraph, ""]
    (DATA_DIR / "w11_rq_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def build_pdf(comparison: pd.DataFrame, metadata: dict, paragraphs: list[str]) -> Path:
    styles = build_styles()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="W11 - Laporan Hasil Eksperimen",
        author="Kelompok Hybrid Fusion (Track G)",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(2.0 * cm, 1.0 * cm, "W11 - Laporan Hasil Eksperimen | Hybrid Fusion (Track G)")
        canvas.drawRightString(A4[0] - 2.0 * cm, 1.0 * cm, f"hal. {document.page}")
        canvas.restoreState()

    protocol = metadata["protocol"]
    story: list = [
        Paragraph("Laporan Hasil Eksperimen W11", styles["title"]),
        Paragraph(
            "Hybrid Fusion Strategies for Collaborative and Content-Based Recommender Systems &bull; "
            "Amazon Reviews 2023 - Musical Instruments &bull; Track G &bull; "
            f"dijalankan {metadata['timestamp_utc'][:19].replace('T', ' ')} UTC",
            styles["subtitle"],
        ),
    ]

    story += [
        Paragraph("1. Ringkasan eksekutif", styles["h1"]),
        Paragraph(paragraphs[0], styles["body"]),
        Paragraph(
            f"Eksperimen menjalankan {metadata['n_configs']} konfigurasi pada validation warm: 3 baseline, "
            f"{len(metadata['search_space']['alphas'])} konfigurasi Weighted Score, dan "
            f"{len(metadata['search_space']['rrf_ks'])} konfigurasi RRF. Model dipilih berdasarkan NDCG@10 "
            "dengan Recall@10 sebagai tie-breaker. Popularity tetap menjadi model terbaik menurut metrik "
            "utama, sedangkan RRF adalah yang terbaik di antara model fusi dan unggul pada Recall@10.",
            styles["body"],
        ),
        Paragraph("2. Protokol dan reproducibility", styles["h1"]),
        Paragraph(
            f"Split yang dipakai adalah official temporal split tanpa pengacakan: training "
            f"{protocol['train_interactions']:,} interaksi dan validation warm "
            f"{protocol['validation_warm_rows']:,} baris. Evaluasi mencakup "
            f"{protocol['evaluable_users']:,} evaluable user pada candidate universe resmi "
            f"{protocol['candidate_universe']:,} item, dengan ranking Top-{protocol['top_n_ranking']} dan "
            f"penilaian pada K = 10, 20, 50. Item relevan adalah rating &gt;= {protocol['positive_threshold']:.1f} "
            f"({protocol['relevant_targets']:,} baris target) dan item yang sudah dilihat saat training "
            "dikeluarkan. Seed 42, metrik utama NDCG@10, dan test set tidak dipakai.",
            styles["body"],
        ),
        Paragraph(
            f"Git commit <font face='Courier'>{metadata['git']['commit'][:12]}</font> "
            f"({'dirty' if metadata['git']['dirty'] else 'clean'}); Python {metadata['python']}; "
            f"pandas {metadata['packages']['pandas']}; scikit-learn {metadata['packages']['scikit-learn']}; "
            f"cache ranking <font face='Courier'>{metadata['cache']['cache_key']}</font>. "
            "Seluruh parameter, hash data, hash cache, dan audit tersimpan di "
            "<font face='Courier'>outputs/w11/w11_metadata.json</font>.",
            styles["body"],
        ),
        Paragraph("3. Hasil utama", styles["h1"]),
        Paragraph(
            "Tabel 1. Metrik validation untuk baseline dan konfigurasi fusi terbaik "
            "(macro average atas 11.659 user).",
            styles["caption"],
        ),
        header_table(comparison),
        Spacer(1, 8),
        Paragraph(
            "Tabel 2. Perbandingan terhadap baseline, termasuk delta relatif, hit rate, dan cakupan katalog.",
            styles["caption"],
        ),
        baseline_table(comparison),
        Spacer(1, 6),
        Paragraph(
            "Pada K yang lebih besar, Popularity tetap unggul NDCG, sedangkan RRF mendekat dan pada "
            "Recall@50 selisihnya menyempit. Pola ini konsisten dengan sifat fusi: ia menambah cakupan, "
            "bukan memperbaiki posisi teratas.",
            styles["body"],
        ),
        PageBreak(),
        Paragraph("4. Visualisasi", styles["h1"]),
    ]

    figures = [
        ("fig1_ndcg_by_model.png", "Gambar 1. NDCG@10/20/50 per model pada validation."),
        ("fig2_metric_vs_k.png", "Gambar 2. Pengaruh nilai K terhadap NDCG dan Recall."),
        ("fig3_search_space_heatmap.png",
         "Gambar 3. Hasil search space: alpha Weighted Score dan konstanta k RRF."),
        ("fig4_user_segments.png", "Gambar 4. NDCG@10 per segmen aktivitas training user."),
        ("fig5_first_hit_position.png", "Gambar 5. Distribusi posisi hit pertama di Top-10."),
    ]
    for index, (name, caption) in enumerate(figures):
        story.append(KeepTogether([figure_flowable(name), Paragraph(caption, styles["caption"])]))
        if index == 2:
            story.append(PageBreak())

    story += [
        Paragraph("5. Analisis hasil dan jawaban RQ", styles["h1"]),
        Paragraph(
            "<b>RQ:</b> apakah fusi hybrid (Weighted Score / RRF) antara Collaborative Filtering dan "
            "Content-Based Filtering menghasilkan Recall@10 dan NDCG@10 yang lebih baik daripada baseline "
            "Popularity, Item-CF, dan TF-IDF CBF?",
            styles["body"],
        ),
    ]
    for paragraph in paragraphs[1:]:
        story.append(Paragraph(paragraph, styles["body"]))

    story += [
        Paragraph("6. Audit dan keterbatasan", styles["h1"]),
    ]
    audit_rows = [["Pemeriksaan", "Status", "Bukti"]]
    for check in metadata["audit"]:
        audit_rows.append([check["pemeriksaan"], check["status"], check["bukti"]])
    audit_table = Table(audit_rows, colWidths=[5.4 * cm, 1.7 * cm, 9.4 * cm])
    audit_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9cfe0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story += [audit_table, Spacer(1, 6)]
    story += [
        Paragraph(
            "Keterbatasan yang perlu dinyatakan di laporan: ranking komponen berasal dari cache Task 3 "
            "(tanpa fitting ulang), search space hanya alpha dan k RRF sesuai CP3, Popularity belum menjadi "
            "komponen fusi, Item-CF masih memakai fallback popularitas, dan belum ada uji signifikansi. "
            "Test set tidak tersedia dan tidak dipakai, sehingga semua angka adalah hasil pengembangan.",
            styles["body"],
        ),
        Paragraph("7. Evidence", styles["h1"]),
        Paragraph(
            "Runner: <font face='Courier'>src/w11_experiment.py</font> &bull; laporan builder: "
            "<font face='Courier'>src/w11_build_report.py</font> &bull; log konfigurasi: "
            "<font face='Courier'>outputs/w11/w11_all_configs.csv</font> &bull; metadata dan audit: "
            "<font face='Courier'>outputs/w11/w11_metadata.json</font> &bull; tabel: "
            "<font face='Courier'>outputs/w11/w11_model_comparison.csv</font>, "
            "<font face='Courier'>w11_user_segments.csv</font> &bull; figure: "
            "<font face='Courier'>outputs/w11/figures/</font> &bull; ringkasan otomatis: "
            "<font face='Courier'>outputs/w11/w11_experiment_summary.md</font> &bull; teks analisis: "
            "<font face='Courier'>outputs/w11/w11_rq_analysis.md</font>.",
            styles["body"],
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF_PATH


def main() -> int:
    comparison, _configs, segments, metadata = load_artifacts()
    paragraphs = analysis_paragraphs(comparison, segments)
    build_markdown(comparison, paragraphs, metadata)
    path = build_pdf(comparison, metadata, paragraphs)
    print("pdf:", path, "| size:", path.stat().st_size, "bytes")
    print("markdown:", MD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
