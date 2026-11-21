# CP3 3.4 - Pemeriksaan dan Analisis Hasil Awal

Dihasilkan otomatis dari artefak Task 2 dan Task 3 oleh `src/task4_initial_analysis.py`.

- Pengguna evaluasi: 11,659
- Metrik: Recall@10 dan NDCG@10 (macro per pengguna)
- Ambang relevan: rating >= 4.0
- Data test dipakai: tidak

## 1. Perbandingan performa

| Model | Recall@10 | NDCG@10 | Selisih recall vs Popularity | Pengguna dengan hit | Item unik di Top-10 |
| --- | --- | --- | --- | --- | --- |
| RRF (k=100) | 0.014198 | 0.009086 | +5.19% | n/a | n/a |
| Popularity | 0.013499 | 0.009222 | +0.00% | 362 | 18 |
| Weighted Score (CF=0.3) | 0.012820 | 0.007930 | -5.02% | n/a | n/a |
| TF-IDF CBF | 0.012234 | 0.007492 | -9.37% | 248 | 16838 |
| Item-CF | 0.005885 | 0.003351 | -56.40% | 128 | 18121 |

## 2. Terkuat dan terlemah

- Terkuat Recall@10: **Popularity**
- Terkuat NDCG@10: **Popularity**
- Terlemah Recall@10: **Item-CF** (rasio 0.436 dari yang terkuat)
- Terlemah NDCG@10: **Item-CF**

## 3. Temuan dan anomali

1. **Nilai Item-CF berbeda antara artefak Task 2 dan Task 3**
   - recall@10 = 0.0058853 di baseline_comparison.csv vs 0.0057995 di task3_model_comparison.csv (selisih 8.58e-05, 1.46% relatif); ndcg@10 = 0.0033510 di baseline_comparison.csv vs 0.0033230 di task3_model_comparison.csv (selisih 2.79e-05, 0.83% relatif). Kemungkinan cache rekomendasi dipakai ulang setelah kode berubah; hapus outputs/task3/cache lalu jalankan ulang sebelum angka dikutip.
   - Dampak: angka CF/CBF belum bisa dikutip sebagai satu nilai final

2. **Nilai TF-IDF CBF berbeda antara artefak Task 2 dan Task 3**
   - recall@10 = 0.0122340 di baseline_comparison.csv vs 0.0121482 di task3_model_comparison.csv (selisih 8.58e-05, 0.70% relatif); ndcg@10 = 0.0074921 di baseline_comparison.csv vs 0.0074909 di task3_model_comparison.csv (selisih 1.20e-06, 0.02% relatif). Kemungkinan cache rekomendasi dipakai ulang setelah kode berubah; hapus outputs/task3/cache lalu jalankan ulang sebelum angka dikutip.
   - Dampak: angka CF/CBF belum bisa dikutip sebagai satu nilai final

3. **RRF unggul Recall@10 tetapi kalah NDCG@10 dari Popularity**
   - Recall@10 0.014198 vs 0.013499 (+5.19%), NDCG@10 0.009086 vs 0.009222 (-1.47%). Fusi menemukan lebih banyak item relevan, tetapi menempatkannya lebih rendah.
   - Dampak: pemilihan model tidak bisa hanya memakai satu metrik

4. **Weighted Score terbaik masih di bawah Popularity**
   - Weighted Score (CF=0.3): Recall@10 0.012820 vs 0.013499. Bobot 30% untuk Item-CF menarik kualitas ranking turun karena komponen CF adalah baseline terlemah.
   - Dampak: bobot optimal cenderung menekan komponen CF, bukan memanfaatkannya

5. **Popularity hanya memakai belasan item untuk seluruh pengguna**
   - Seluruh 11,659 pengguna hanya menerima 18 item unik di Top-10, dan hanya 6 item berbeda di peringkat 1. Item-CF menyentuh 18,121 item, TF-IDF CBF 16,838 item.
   - Dampak: keunggulan recall Popularity dicapai dengan portfolio hampir seragam

6. **Mayoritas pengguna hanya punya satu item relevan**
   - 54.52% pengguna memiliki tepat satu target relevan, rata-rata 2.08 target per pengguna. Untuk kelompok ini Recall@10 bersifat biner dan NDCG@10 hanya ditentukan posisi satu item.
   - Dampak: perbedaan model kecil sulit dibedakan secara statistik

### Irisan pengguna yang mendapat minimal satu hit

| Kombinasi model | Jumlah pengguna |
| --- | --- |
| tidak ada hit | 10984 |
| Popularity | 333 |
| TF-IDF CBF | 207 |
| Item-CF | 73 |
| Item-CF + TF-IDF CBF | 33 |
| Item-CF + Popularity | 21 |
| Popularity + TF-IDF CBF | 7 |
| Item-CF + Popularity + TF-IDF CBF | 1 |

### Perbandingan berpasangan (Recall@10)

| A | B | A lebih baik | B lebih baik | Seri |
| --- | --- | --- | --- | --- |
| Item-CF | Popularity | 106 | 340 | 11213 |
| Item-CF | TF-IDF CBF | 95 | 216 | 11348 |
| Popularity | TF-IDF CBF | 354 | 241 | 11064 |

## 4. Insight untuk tahap berikutnya

1. Item-CF adalah baseline terlemah: Recall@10 0.005885, sekitar 56% di bawah Popularity, dan hanya 128 pengguna mendapat satu pun item relevan. Fallback popularitas di recommend_from_model membuat Top-10 Item-CF sebagian besar bukan hasil kemiripan item, jadi kontribusi CF murni perlu diukur terpisah.
2. Ketiga baseline hampir tidak beririsan: hanya 1 pengguna mendapat hit di ketiganya, sedangkan 333 pengguna hanya tertolong Popularity, 207 hanya TF-IDF CBF, dan 73 hanya Item-CF. Ini menjelaskan mengapa fusi berbasis ranking bisa menaikkan Recall di atas semua komponennya.
3. Prioritas perbaikan berikutnya adalah memperbaiki skor Item-CF (normalisasi jumlah tetangga, ambang co-occurrence minimum, dan memisahkan fallback popularitas sebagai varian tersendiri) sebelum menambah model baru seperti MF.
4. Popularity sebaiknya ikut dijadikan komponen fusi, bukan hanya pembanding: karena ia menang pada pengguna yang tidak tertolong CF/CBF, daftar ketiga berpeluang memperbaiki NDCG fusi yang saat ini justru lebih rendah dari Popularity.
5. Pelaporan perlu ditambah hit-rate@10 dan catalog coverage. Popularity hanya menyentuh 18 item unik, sementara 3.10% pengguna mendapat hit; angka recall saja menyembunyikan perbedaan cakupan ini.
6. Sebelum angka CF/CBF dikutip di laporan, hapus outputs/task3/cache lalu jalankan ulang Task_3_Evaluasi_Eksperimen_Hybrid_FAST supaya nilainya sama dengan baseline_comparison.csv.
