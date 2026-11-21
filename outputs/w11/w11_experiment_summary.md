# W11 - Laporan Hasil Eksperimen (validation)

Dihasilkan otomatis oleh `src/w11_experiment.py` pada 2026-11-21T15:51:11.467508+00:00.

## 1. Protokol

- Split: official temporal; training 427,957 interaksi, validation warm 27,537 baris
- Evaluable users: 11,659
- Candidate universe: 22,753 item; ranking Top-200, dinilai Top-10/20/50
- Relevan: rating >= 4.0; target setelah filter: 24,202 baris
- Metrik utama pemilihan: NDCG@10 (tie-breaker Recall@10)
- Test set dipakai: tidak
- Seed: 42; commit: e7ac087530 (dirty)

## 2. Tabel hasil utama (model x metrik)

| Model | Konfigurasi | NDCG@10 | NDCG@20 | NDCG@50 | Recall@10 | Recall@20 | Recall@50 | Precision@10 | MRR@10 | MAP@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | E1-popularity | 0.009222 | 0.011476 | 0.016235 | 0.013499 | 0.021493 | 0.042006 | 0.003182 | 0.012988 | 0.005490 |
| Item-CF | E2-itemcf-k100 | 0.003323 | 0.004366 | 0.006495 | 0.005800 | 0.009353 | 0.018193 | 0.001141 | 0.003536 | 0.001981 |
| TF-IDF CBF | E3-tfidf-cbf | 0.007491 | 0.009546 | 0.013057 | 0.012148 | 0.019233 | 0.034442 | 0.002196 | 0.008095 | 0.004955 |
| Weighted Score | E4-weighted-a030 | 0.007930 | 0.009973 | 0.013443 | 0.012820 | 0.019631 | 0.034679 | 0.002281 | 0.008818 | 0.005222 |
| RRF | E5-rrf-k100 | 0.009086 | 0.011097 | 0.014563 | 0.014198 | 0.020977 | 0.035764 | 0.002873 | 0.010687 | 0.005873 |

## 3. Perbandingan terhadap baseline

| Model | NDCG@10 | Delta NDCG vs Popularity | Recall@10 | Delta Recall vs Popularity |
| --- | --- | --- | --- | --- |
| Popularity | 0.009222 | +0.00% | 0.013499 | +0.00% |
| Item-CF | 0.003323 | -63.97% | 0.005800 | -57.04% |
| TF-IDF CBF | 0.007491 | -18.77% | 0.012148 | -10.00% |
| Weighted Score | 0.007930 | -14.01% | 0.012820 | -5.02% |
| RRF | 0.009086 | -1.47% | 0.014198 | +5.19% |

Model terbaik keseluruhan menurut NDCG@10: **E1-popularity** (NDCG@10 0.009222, Recall@10 0.013499).

## 4. Temuan

1. Popularity tetap terkuat pada NDCG@10 (0.009222) dan MRR@10 (0.012988); fusi tidak mengalahkannya pada metrik berbobot posisi.
2. RRF (k=100) unggul Recall@10 (0.014198, +5.19% vs Popularity) tetapi NDCG@10 0.009086 (-1.47%).
3. Weighted Score terbaik alpha=0.3: NDCG@10 0.007930, Recall@10 0.012820; masih di bawah Popularity pada kedua metrik.
4. Item-CF terlemah: NDCG@10 0.003323, Recall@10 0.005800 (-57.04% vs Popularity).
5. TF-IDF CBF: NDCG@10 0.007491, Recall@10 0.012148, dengan cakupan katalog jauh lebih luas (16,851 item unik di Top-10 vs 18 item untuk Popularity).
6. Search space: 20 konfigurasi dievaluasi (11 Weighted Score, 6 RRF, 3 baseline).

## 5. Segmen aktivitas user (NDCG@10)

| Segmen | E1-popularity | E2-itemcf-k100 | E3-tfidf-cbf | E4-weighted-a000 | E4-weighted-a030 | E4-weighted-a100 | E5-rrf-k100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-4 | 0.011089 | 0.003016 | 0.007838 | 0.007838 | 0.008065 | 0.003042 | 0.009105 |
| 5-9 | 0.008564 | 0.003214 | 0.008390 | 0.008420 | 0.008748 | 0.003202 | 0.008966 |
| 10-19 | 0.007076 | 0.003811 | 0.004476 | 0.004476 | 0.005441 | 0.003811 | 0.008697 |
| 20+ | 0.006705 | 0.004543 | 0.008129 | 0.008129 | 0.008982 | 0.004524 | 0.010561 |

Jumlah user per segmen: 1-4: 4772, 5-9: 4142, 10-19: 1953, 20+: 792

## 6. Audit

| Pemeriksaan | Status | Bukti |
| --- | --- | --- |
| protokol ranking - Popularity | PASS | 11659 user tersedia, 11659 dievaluasi; max 200 item; 0 duplikat; 0 di luar kandidat; 0 seen-item |
| protokol ranking - Item-CF | PASS | 11659 user tersedia, 11659 dievaluasi; max 200 item; 0 duplikat; 0 di luar kandidat; 0 seen-item |
| protokol ranking - TF-IDF CBF | PASS | 11659 user tersedia, 11659 dievaluasi; max 200 item; 0 duplikat; 0 di luar kandidat; 0 seen-item |
| target relevan tidak training-seen | PASS | 0 overlap |
| user universe sama untuk semua model | PASS | 11659 user per model |

## 7. Figure

- `d:/Semester 7/Sisrek/Praktikum/proyek/outputs/w11/figures/fig1_ndcg_by_model.png`
- `d:/Semester 7/Sisrek/Praktikum/proyek/outputs/w11/figures/fig2_metric_vs_k.png`
- `d:/Semester 7/Sisrek/Praktikum/proyek/outputs/w11/figures/fig3_search_space_heatmap.png`
- `d:/Semester 7/Sisrek/Praktikum/proyek/outputs/w11/figures/fig4_user_segments.png`
- `d:/Semester 7/Sisrek/Praktikum/proyek/outputs/w11/figures/fig5_first_hit_position.png`

## 8. Keterbatasan

- Ranking komponen dimuat dari cache Task 3 (hash tercatat di metadata), sehingga metrik mereproduksi run tersebut; tidak ada fitting ulang pada W11.
- Test set belum tersedia dan tidak dipakai; seluruh angka adalah hasil pengembangan.
- Search space terbatas pada alpha dan k RRF sesuai CP3; tidak ada model baru.
- Item-CF masih memakai fallback popularitas untuk skor nol.
