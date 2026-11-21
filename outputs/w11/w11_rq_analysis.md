# W11 - Analisis Hasil untuk Menjawab RQ

Semua angka dari `outputs/w11/` (validation warm, 11,659 user, test tidak dipakai). Metrik utama pemilihan: NDCG@10 sesuai CP3; K tambahan 20 dan 50.

| Model | NDCG@10 | Recall@10 | NDCG@20 | NDCG@50 | Recall@50 |
| --- | --- | --- | --- | --- | --- |
| Popularity | 0.009222 | 0.013499 | 0.011476 | 0.016235 | 0.042006 |
| Item-CF | 0.003323 | 0.005800 | 0.004366 | 0.006495 | 0.018193 |
| TF-IDF CBF | 0.007491 | 0.012148 | 0.009546 | 0.013057 | 0.034442 |
| Weighted Score | 0.007930 | 0.012820 | 0.009973 | 0.013464 | 0.034782 |
| RRF | 0.009086 | 0.014198 | 0.011097 | 0.014563 | 0.035764 |

## Analisis

Temuan utama. Fusi hybrid berhasil menaikkan Recall@10 di atas seluruh baseline, tetapi tidak pada NDCG@10. RRF (k=100) mencapai Recall@10 0.014198 (+5.19% dibanding Popularity), sementara NDCG@10 0.009086 masih -1.47% di bawah Popularity. Artinya, fusi menemukan lebih banyak item relevan, tetapi menempatkannya pada posisi yang lebih rendah.

a. Apakah track mengungguli baseline? Ya, tetapi hanya pada Recall@10. RRF unggul +5.19% atas Popularity, +16.88% atas TF-IDF CBF, dan +144.82% atas Item-CF. Pada NDCG@10 RRF kalah -1.47% dari Popularity, meskipun tetap +21.30% di atas TF-IDF CBF dan +173.44% di atas Item-CF. Weighted Score terbaik (alpha=0,3) tidak mengungguli Popularity pada kedua metrik, tetapi memberi perbaikan kecil atas komponen CBF-nya sendiri: +5.87% NDCG@10 dan +5.53% Recall@10.

b. Kapan model bekerja baik atau buruk? Popularity paling kuat pada user dengan sedikit interaksi (segmen 1-4: NDCG@10 0.011089) dan melemah pada user aktif (20+: 0.006705). RRF justru paling stabil dan tertinggi pada user aktif (20+: 0.010561; 10-19: 0.008697), karena fusi memanfaatkan riwayat yang lebih panjang. TF-IDF CBF dan Weighted Score turun tajam pada segmen 10-19 (0.004476 dan 0.005441) lalu pulih pada 20+. Item-CF terlemah di semua segmen, walau membaik seiring aktivitas (0.003016 pada 1-4 menjadi 0.004543 pada 20+).

c. Apakah hasil mendukung RQ? Sebagian. RQ menanyakan apakah fusi menghasilkan Recall@10 dan NDCG@10 yang lebih baik daripada Popularity, Item-CF, dan TF-IDF CBF. Untuk Recall@10, jawabannya ya dan konsisten dengan Tabel 2. Untuk NDCG@10, jawabannya belum: Popularity tetap tertinggi (0.009222), diikuti RRF (0.009086), Weighted Score (0.007930), TF-IDF CBF (0.007491), dan Item-CF (0.003323). Hipotesis bahwa fusi selalu lebih baik tidak didukung data ini.

d. Temuan tak terduga. Popularity hanya memakai 18 item unik di seluruh Top-10, tetapi memenangi NDCG@10 dan MRR@10 (0.012988 vs 0.010687 untuk RRF), tanda bahwa sinyal personal pada data ini lemah dan posisi teratas didominasi item populer. Sebaliknya, keunggulan Recall RRF datang dari cakupan katalog yang jauh lebih luas (18,897 item unik vs 18). Sensitivitas bobot juga tajam: NDCG@10 memuncak pada alpha=0,3 (0,007930) lalu merosot saat bobot Item-CF dinaikkan (alpha=0,5: 0,007411; alpha=0,7: 0,005039; alpha=1,0: 0,003338). Untuk RRF, NDCG@10 naik monoton terhadap k dan terbaik pada k=100 (0,009086).

e. Keterbatasan. (1) Ranking komponen dimuat dari cache Task 3 yang sudah diaudit, sehingga tidak ada fitting ulang pada W11; angka mereproduksi run tersebut secara persis. (2) Search space terbatas pada alpha dan k RRF sesuai CP3; Popularity belum dijadikan komponen fusi. (3) Item-CF masih memakai fallback popularitas untuk skor nol, sehingga kontribusi CF murni belum terisolasi. (4) Belum ada uji signifikansi, padahal 10.984 dari 11.659 user (94,2%) tidak memperoleh hit dan 54,5% hanya punya satu target. (5) Test set belum tersedia dan tidak dipakai; seluruh angka adalah hasil pengembangan di validation, dan evaluasi final dilakukan di W13.
