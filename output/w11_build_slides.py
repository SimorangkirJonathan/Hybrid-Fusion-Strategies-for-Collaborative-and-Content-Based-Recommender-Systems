"""Generate output/W11_Experiment_Progress_Evidence.pptx (2 slide, terisi).

Semua angka dibaca/dikutip dari artefak Task 2 dan Task 3 yang sudah ada:
- outputs/task3/task3_model_comparison.csv  (angka utama slide 1)
- outputs/task3/task3_summary.json          (parameter & protokol)
- outputs/task3/task3_audit_summary.json    (audit evaluable users)
- outputs/task3/task3_leakage_audit.csv     (tabel audit slide 2)
- outputs/task2/task2_protocol_and_results.json (protokol Task 2)

Jalankan: python output/w11_build_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x1B, 0x26, 0x5E)
BLUE = RGBColor(0x27, 0x45, 0x9B)
ACCENT = RGBColor(0x2F, 0x5B, 0xE0)
ORANGE = RGBColor(0xDF, 0x6B, 0x0A)
RED = RGBColor(0xC0, 0x00, 0x00)
GRAY = RGBColor(0x59, 0x59, 0x59)
DARK = RGBColor(0x1F, 0x1F, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF2, 0xF4, 0xFA)
GREEN = RGBColor(0x1E, 0x7B, 0x34)
FONT = "Calibri"


def textbox(slide, x, y, w, h, content, size=10, bold=False, color=DARK,
            italic=False, align=PP_ALIGN.LEFT, line=1.0, gap=0,
            anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = content if isinstance(content, list) else [[(content, {})]]
    for i, para_runs in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = line
        if gap:
            par.space_after = Pt(gap)
        for txt, opt in para_runs:
            run = par.add_run()
            run.text = txt
            f = run.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
    return box


def hline(slide, x, y, w, color=NAVY, thickness=1.25):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Pt(thickness))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False


def section(slide, x, y, num, title, tw=5.2):
    textbox(slide, x, y, 0.55, 0.32, num, size=17, bold=True, color=ACCENT)
    textbox(slide, x + 0.52, y + 0.03, tw, 0.32, title, size=14.5, bold=True, color=BLUE)


def body(slide, x, y, w, lines, size=9, color=DARK, gap=3.5, prefix="\u2022  "):
    textbox(slide, x, y, w, 1.8, [[(prefix + t, {})] for t in lines],
            size=size, color=color, gap=gap, line=1.05)


def footer(slide, num):
    textbox(slide, 0.55, 7.12, 5.0, 0.22, "1254054 \u2022 Sistem Rekomendasi",
            size=8, color=GRAY)
    textbox(slide, 8.6, 7.12, 4.18, 0.22,
            f"W11 \u2022 Experiment Progress Evidence      {num}",
            size=8, color=GRAY, align=PP_ALIGN.RIGHT)


def make_table(slide, x, y, widths, rows, head_h=0.30, row_h=0.42,
               head_size=9.5, body_size=9, aligns=None, status_color=None,
               status_is_bold=True):
    n_rows, n_cols = len(rows), len(widths)
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(sum(widths)),
                                   Inches(head_h + row_h * (n_rows - 1)))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for j, wdt in enumerate(widths):
        tbl.columns[j].width = Inches(wdt)
    tbl.rows[0].height = Inches(head_h)
    for i in range(1, n_rows):
        tbl.rows[i].height = Inches(row_h)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (WHITE if i % 2 else LIGHT)
            tf = cell.text_frame
            tf.word_wrap = True
            par = tf.paragraphs[0]
            par.alignment = aligns[j] if aligns else PP_ALIGN.LEFT
            par.line_spacing = 1.0
            run = par.add_run()
            run.text = value
            f = run.font
            f.name = FONT
            f.size = Pt(head_size if i == 0 else body_size)
            f.bold = i == 0
            color = WHITE if i == 0 else DARK
            if i > 0 and j == 1 and status_color:
                color = status_color
                f.bold = status_is_bold
            f.color.rgb = color
    return tbl


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------------- SLIDE 1 ----------------
    s1 = prs.slides.add_slide(blank)
    textbox(s1, 0.55, 0.28, 12.2, 0.5, "W11 \u2013 Experiment Progress Evidence",
            size=29, bold=True, color=BLUE)
    textbox(s1, 0.55, 0.86, 12.2, 0.28,
            "Template submission eCourse \u2022 maksimum 2 slide \u2022 "
            "gunakan hasil VALIDATION, bukan FinalTest",
            size=10.5, italic=True, color=GRAY)
    hline(s1, 0.55, 1.21, 12.23, NAVY, 1.5)

    section(s1, 0.55, 1.30, "01", "Identitas Eksperimen")
    section(s1, 7.00, 1.30, "02", "Evaluation Protocol", tw=5.6)

    def field(slide, lx, vx, y, label, value, size=9.5):
        textbox(slide, lx, y, 1.55, 0.9, label, size=10.5, bold=True, color=NAVY)
        textbox(slide, vx, y + 0.01, 4.55 if vx < 7 else 4.40, 1.0, value,
                size=size, color=DARK, line=1.08)

    field(s1, 0.55, 2.20, 1.78, "Kelompok", "G_ \u2022 Track G")
    field(s1, 0.55, 2.20, 2.26, "Track / model",
          "Track G \u2014 Hybrid Fusion: Item-CF (collaborative) + TF-IDF CBF "
          "(content-based), digabung dengan Weighted Score dan Reciprocal Rank "
          "Fusion (RRF). MF belum tersedia.")
    field(s1, 0.55, 2.20, 2.84, "Research Question",
          "Apakah fusi hybrid (Weighted Score / RRF) antara Collaborative "
          "Filtering dan Content-Based Filtering menghasilkan Recall@10 dan "
          "NDCG@10 lebih baik daripada baseline Popularity, Item-CF, dan "
          "TF-IDF CBF?")

    field(s1, 7.00, 8.40, 1.78, "Split / data",
          "Official temporal split (tanpa random split): train 427.957, "
          "validation 41.284. Evaluasi memakai subset warm-start "
          "validation_warm: 11.659 user, 27.537 interaksi. Test tidak dipakai.")
    field(s1, 7.00, 8.40, 2.52, "Relevance",
          "Rating \u2265 4,0 = relevan (protokol Task 1); 24.202 baris target, "
          "item yang sudah dilihat dibuang.")
    field(s1, 7.00, 8.40, 3.06, "Candidate set",
          "Universe resmi 22.753 item (candidate_items_warm); ranking Top-200, "
          "dinilai Top-10.")
    field(s1, 7.00, 8.40, 3.48, "Metrics",
          "Recall@10 dan NDCG@10, macro average 11.659 user.")

    section(s1, 0.55, 3.98, "03", "Current Experimental Results", tw=6.0)
    textbox(s1, 7.30, 4.02, 5.48, 0.30,
            "Hanya hasil yang benar-benar tersedia. Jika belum ada, tulis "
            "\u201cNot available\u201d + alasan singkat.",
            size=8, italic=True, color=GRAY)

    rows = [
        ["Model", "Recall@10", "NDCG@10", "Status / Notes"],
        ["Popularity", "0,01350", "0,00922",
         "Done. 362/11.659 user hit; hanya 18 item unik di Top-10."],
        ["Collaborative baseline (Item-CF)", "0,00580", "0,00332",
         "Done. Terlemah: \u221256,4% Recall vs Popularity; 128 user hit."],
        ["Additional baseline (TF-IDF CBF)", "0,01215", "0,00749",
         "Done. \u22129,4% Recall vs Popularity; 16.838 item unik."],
        ["Track / proposed \u2014 RRF (k=100)", "0,01420", "0,00909",
         "Done. Recall +5,19% vs Popularity, NDCG \u22121,47%. "
         "Weighted Score (CF=0,3): 0,01282 / 0,00793."],
    ]
    make_table(s1, 0.55, 4.42, [4.35, 2.15, 2.15, 3.58], rows,
               aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT],
               body_size=8.5, row_h=0.44)

    textbox(s1, 0.55, 6.56, 12.23, 0.28,
            [[("Minimum target hari ini", {"bold": True, "color": ORANGE}),
              (" \u2014 minimal satu baseline/model jalan end-to-end di "
               "validation sampai menghasilkan Recall@10 dan NDCG@10.  "
               "Status: TERCAPAI (5 model dievaluasi).", {"color": DARK})]],
            size=9.5)
    textbox(s1, 1.60, 6.92, 10.2, 0.26,
            "Jangan gunakan FinalTest untuk memilih model atau melakukan tuning.",
            size=9, bold=True, color=RED, align=PP_ALIGN.CENTER)
    footer(s1, 1)

    # ---------------- SLIDE 2 ----------------
    s2 = prs.slides.add_slide(blank)
    textbox(s2, 0.55, 0.28, 12.2, 0.5, "Evidence: Protokol & Bukti Audit",
            size=27, bold=True, color=BLUE)
    textbox(s2, 0.55, 0.82, 12.2, 0.28,
            "Lampiran slide 1 \u2022 seluruh angka berasal dari artefak Task 2 & "
            "Task 3 \u2022 validation only, tanpa FinalTest",
            size=10, italic=True, color=GRAY)
    hline(s2, 0.55, 1.16, 12.23, NAVY, 1.5)

    section(s2, 0.55, 1.24, "01", "Data & Split", tw=4.6)
    body(s2, 0.55, 1.60, 5.35, [
        "Official split: train 427.957 \u2022 validation 41.284 (warm 30.360)",
        "Evaluable users: 11.659 \u2192 27.537 interaksi validation warm",
        "Candidate universe: 22.753 item \u2022 Top-200 per model \u2022 dinilai Top-10",
    ])

    section(s2, 0.55, 2.28, "02", "Ground Truth & Exclusion", tw=4.8)
    body(s2, 0.55, 2.64, 5.35, [
        "Relevan = rating \u2265 4,0: 24.202 baris, 11.659 user, 2,08 target/user",
        "54,52% user punya 1 target \u2192 Recall biner untuk mayoritas user",
        "Seen-item exclusion aktif: 0 overlap target\u2013riwayat training",
    ])

    section(s2, 0.55, 3.32, "03", "Konfigurasi & Aturan Pemilihan", tw=5.4)
    body(s2, 0.55, 3.68, 5.35, [
        "Item-CF: biner positif, cosine, top-100 neighbor",
        "CBF: TF-IDF 1\u20132 gram, min_df=2, max_df=0,8, 100k fitur, top-100",
        "Fusi: Weighted \u03b1=0,3 \u2022 RRF k=100 \u2022 seleksi via NDCG@10",
        "Fallback skor nol = urutan popularitas train; skor bukan probabilitas",
    ])

    section(s2, 0.55, 4.66, "04", "Catatan Konsistensi Angka", tw=5.2)
    body(s2, 0.55, 5.02, 5.35, [
        "Task 2 vs Task 3: Item-CF 0,00589 vs 0,00580 karena cache rekomendasi",
        "Slide 1 mengutip satu sumber (Task 3) agar semua baris sebanding",
        "MF belum diimplementasikan \u2192 tidak dilaporkan sebagai hasil",
    ])

    section(s2, 6.55, 1.24, "05", "Audit \u2014 task3_leakage_audit.csv", tw=5.9)
    audit = [
        ["Pemeriksaan", "Status", "Bukti"],
        ["Official evaluable users digunakan", "PASS", "11.659 user"],
        ["Semua evaluable users ada di training", "PASS", "0 missing"],
        ["Validation user di evaluable universe", "PASS", "0 outside"],
        ["Seen-item exclusion pada target", "PASS", "0 overlap"],
        ["User universe sama untuk semua model", "PASS", "11.659 user/model"],
        ["src model/fusion bebas baca validation/test", "PASS", "static scan"],
        ["Notebook fit-call bebas arg validation/test", "PASS", "9 fit calls"],
        ["Test file tersedia", "NOT AVAILABLE", "tidak tersedia"],
        ["Matrix Factorization", "INFO", "belum ada di repo"],
    ]
    make_table(s2, 6.55, 1.60, [3.45, 1.30, 1.48], audit,
               aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT],
               body_size=7.5, head_size=8.5, row_h=0.33, status_color=GREEN)
    textbox(s2, 6.55, 4.68, 6.23, 0.28,
            "PASS = pemeriksaan otomatis pada skrip audit Task 3 lolos "
            "(tanpa kebocoran validation/test ke tahap fitting).",
            size=8, italic=True, color=GRAY)

    textbox(s2, 0.55, 5.94, 12.23, 0.28,
            "Status W11: pipeline lengkap \u2014 5 model (3 baseline + 2 fusi) "
            "berjalan end-to-end di validation; target minimum tercapai.",
            size=10.5, bold=True, color=NAVY)
    textbox(s2, 0.55, 6.30, 12.23, 0.50,
            "Sumber angka: outputs/task3/task3_model_comparison.csv "
            "(Task_3_Evaluasi_Eksperimen_Hybrid_FAST.ipynb) \u2022 parameter: "
            "task3_summary.json \u2022 audit: task3_audit_summary.json, "
            "task3_leakage_audit.csv \u2022 split resmi: "
            "Task_1_Experimental_Protocol.ipynb (audit 427.957 / 41.284) "
            "\u2022 protokol Task 2: task2_protocol_and_results.json",
            size=8.5, color=GRAY)
    textbox(s2, 0.55, 6.70, 12.23, 0.30,
            "Catatan: auditor dapat memeriksa ulang angka dengan menjalankan "
            "notebook Task 3 (cache di outputs/task3/cache) dan Task 4 "
            "(hanya membaca CSV, tidak menghitung ulang metrik).",
            size=8.5, italic=True, color=GRAY)
    footer(s2, 2)

    out = Path("output") / "W11_Experiment_Progress_Evidence.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print("saved:", path, "| size:", path.stat().st_size, "bytes")
