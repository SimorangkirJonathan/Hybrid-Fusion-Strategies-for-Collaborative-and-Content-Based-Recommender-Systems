"""Generate output/W11_Preliminary_Findings_Next_Action.pptx (1 slide).

Halaman 05 (Preliminary Findings) + 06 (Next Action menuju 21 November)
direkonstruksi dari screenshot template. Semua angka dikutip dari:
- outputs/task3/task3_model_comparison.csv   (Recall@10 / NDCG@10)
- outputs/task3/task3_summary.json           (protokol, universe 22.753)
- outputs/task4/task4_initial_analysis.json  (hit users, coverage, irisan)

Deck sengaja terpisah dari W11_Experiment_Progress_Evidence.pptx supaya
batas "maksimum 2 slide" untuk submission evidence tidak dilanggar.

Jalankan: python output/w11_build_findings_slide.py
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w11_build_slides import (  # noqa: E402
    ACCENT, DARK, GRAY, LIGHT, NAVY, ORANGE, footer, hline, make_table,
    section, textbox,
)

# ---------------------------------------------------------------- angka kunci
POP_R, POP_N = 0.013499, 0.009222
CF_R, CF_N = 0.005800, 0.003323
CBF_R, CBF_N = 0.012148, 0.007491
WS_R, WS_N = 0.012820, 0.007930
RRF_R, RRF_N = 0.014198, 0.009086
TARGETS = 24202
USERS = 11659
UNIVERSE = 22753
RULE = RGBColor(0xC9, 0xCF, 0xE0)


def dec(x, digits=5):
    return f"{x:.{digits}f}".replace(".", ",")


def thou(n):
    return f"{n:,}".replace(",", ".")

FINDINGS = [
    ("01", "Temuan utama dari perbandingan model",
     f"RRF (k=100) memberi Recall@10 tertinggi: {dec(RRF_R)} vs Popularity {dec(POP_R)} "
     f"(+5,19%), TF-IDF CBF {dec(CBF_R)} (+16,9%), Item-CF {dec(CF_R)} (2,45\u00d7 lebih rendah). "
     f"Namun NDCG@10 RRF {dec(RRF_N)} masih di bawah Popularity {dec(POP_N)} (\u22121,47%). "
     f"Weighted Score (CF=0,3): {dec(WS_R)} / {dec(WS_N)}, masih di bawah Popularity."),
    ("02", "Apa arti perubahan Recall@10?",
     f"Recall@10 = rata-rata proporsi target relevan per user yang masuk Top-10 "
     f"(rata-rata {dec(TARGETS / USERS, 2)} target/user). Selisih +{dec(RRF_R - POP_R)} "
     f"RRF vs Popularity setara \u2248{round((RRF_R - POP_R) * TARGETS)} hit tambahan dari "
     f"{thou(TARGETS)} target \u2014 arahnya benar, tetapi besarnya kecil dan belum diuji "
     f"signifikansinya."),
    ("03", "Apa arti perubahan NDCG@10?",
     "NDCG@10 memberi bobot lebih besar pada posisi atas (diskon log2). RRF \u22121,47% "
     "berarti target yang ditemukan RRF lebih sering mendarat di posisi 4\u201310, bukan 1\u20133; "
     "untuk 54,5% user yang hanya punya 1 target, pergeseran posisi ini langsung menurunkan skor."),
    ("04", "Unexpected result / issue yang perlu diverifikasi",
     f"(a) Item-CF justru terlemah ({dec(CF_R)}, hanya {CF_R / (10 / UNIVERSE):.0f}\u00d7 di atas "
     "acak) pada matriks sangat sparse (densitas 0,034%); (b) Popularity hanya memakai 18 item "
     f"unik untuk {thou(USERS)} user tetapi memenangi NDCG; (c) 10.984 user (94,2%) tidak dapat "
     "hit dari model mana pun; (d) angka Item-CF/CBF beda 0,7\u20131,5% antara Task 2 dan Task 3 "
     "akibat cache \u2192 harus dijalankan ulang."),
    ("05", "Indikasi jawaban sementara terhadap RQ",
     "Sementara: YA untuk Recall@10 (RRF +5,19% di atas baseline terbaik) tetapi BELUM untuk "
     "NDCG@10 (\u22121,47%). Jadi fusi baru unggul pada satu metrik; klaim \u201cfusi lebih baik\u201d "
     "belum bisa dibuat sebelum komponen CF diperbaiki dan selisihnya diuji signifikansi."),
]

ACTIONS = [
    ["Action", "PIC", "Target"],
    ["Perbaiki skor Item-CF (normalisasi tetangga, ambang co-occurrence) + hapus cache "
     "Task 3 lalu jalankan ulang sampai satu set angka final", "[Nama]", "18 Nov"],
    ["Uji signifikansi RRF vs Popularity (bootstrap/paired) + tambahkan hit-rate@10 dan "
     "catalog coverage ke laporan", "[Nama]", "19 Nov"],
    ["Tambahkan Popularity sebagai komponen fusi, tuning bobot & k RRF di validation, "
     "susun draft laporan CP3", "[Nama]", "20 Nov"],
]


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    section(s, 0.55, 0.30, "05", "Preliminary Findings", tw=6.0)
    textbox(s, 7.20, 0.34, 5.58, 0.28,
            "Semua angka = validation warm-start (11.659 user), bukan FinalTest",
            size=8.5, italic=True, color=GRAY, align=PP_ALIGN.RIGHT)
    hline(s, 0.55, 0.70, 12.23, NAVY, 1.25)

    y = 0.84
    step = 0.76
    for num, title, answer in FINDINGS:
        textbox(s, 0.55, y, 0.34, 0.22, num, size=9, bold=True, color=ORANGE)
        textbox(s, 0.92, y - 0.01, 11.86, 0.24, title, size=9.5, bold=True, color=NAVY)
        textbox(s, 0.92, y + 0.20, 11.86, 0.52, answer, size=8.3, color=DARK, line=1.05)
        hline(s, 0.55, y + 0.70, 12.23, RULE, 0.75)
        y += step

    section(s, 0.55, y - 0.02, "06", "Next Action menuju 21 November", tw=6.6)
    textbox(s, 7.20, y + 0.02, 5.58, 0.26,
            "Batas submission 21 November \u2022 semua keputusan tetap dari validation",
            size=8.5, italic=True, color=GRAY, align=PP_ALIGN.RIGHT)
    make_table(s, 0.55, y + 0.36, [7.95, 1.70, 2.58], ACTIONS,
               aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
               head_size=9, body_size=8.3, head_h=0.30, row_h=0.36)
    textbox(s, 0.55, y + 1.98, 12.23, 0.40,
            "Risiko yang perlu dijaga: bila Item-CF tetap lemah setelah perbaikan, "
            "laporkan fusi hanya unggul pada Recall@10 dan jelaskan sebabnya \u2014 "
            "jangan menutupinya dengan tuning di test atau dengan mengganti metrik.",
            size=8.5, italic=True, color=GRAY)

    footer(s, 3)

    out = Path("output") / "W11_Preliminary_Findings_Next_Action.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print("saved:", path, "| size:", path.stat().st_size, "bytes")
