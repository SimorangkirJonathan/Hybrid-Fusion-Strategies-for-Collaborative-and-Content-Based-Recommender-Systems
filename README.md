# Sistem Rekomendasi Musical Instruments

Proyek ini membangun dan membandingkan baseline rekomendasi pada dataset Amazon Reviews 2023 kategori Musical Instruments.

## Isi utama

- `01_eda_training.ipynb` ? eksplorasi data training dan katalog item.
- `Task_1_Experimental_Protocol.ipynb` ? protokol eksperimen dan baseline recommender.
- `Task_2_Candidate_Set_dan_Metrik.ipynb` ? audit candidate set serta implementasi Recall@10 dan NDCG@10.
- `02_baseline_recommenders.ipynb` ? Popularity, collaborative Item-CF, TF-IDF Content-Based Filtering (CBF), dan sanity check rekomendasi.
- `03_hybrid_fusion.ipynb` ? Weighted Score Fusion dan Reciprocal Rank Fusion (RRF) dari ranking Item-CF dan TF-IDF CBF.
- `src/` ? fungsi untuk memuat data, membangun recommender, mengevaluasi ranking, dan menggabungkan ranking.

## Data yang digunakan

Letakkan dua folder dataset di direktori proyek dengan struktur berikut (nama folder boleh memiliki suffix tanggal seperti yang tersedia saat ini):

```text
proyek/
??? MAIN DATA-.../
?   ??? MAIN DATA/
?       ??? interactions_train.csv
?       ??? items.csv
?       ??? candidate_items_warm.csv
??? SUPPORTING_DATA-.../
?   ??? SUPPORTING_DATA/
?       ??? interactions_validation_warm.csv
?       ??? evaluable_users_validation_warm.csv
??? src/
??? 02_baseline_recommenders.ipynb
??? 03_hybrid_fusion.ipynb
```

`interactions_train.csv` menjadi sumber interaksi untuk Popularity dan Item-CF, serta riwayat positif user untuk CBF. TF-IDF mempelajari representasi teks dari metadata `items.csv`; metadata ini adalah fitur konten item, bukan label/rating validation. Validation warm hanya digunakan untuk pemeriksaan contoh, evaluasi, dan pemilihan parameter fusi. Item yang sudah muncul pada riwayat training user dikeluarkan dari rekomendasinya.

## Menjalankan baseline (Windows PowerShell)

Jalankan perintah dari direktori proyek:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m jupyter lab
```

Buka dan jalankan `Task_1_Experimental_Protocol.ipynb`, kemudian `Task_2_Candidate_Set_dan_Metrik.ipynb` untuk mengaudit candidate set dan metrik evaluasi. Setelah itu jalankan `02_baseline_recommenders.ipynb` dan `03_hybrid_fusion.ipynb` untuk baseline serta perbandingan strategi hybrid. Notebook mengharapkan folder `src/` dan kedua folder dataset berada di bawah direktori proyek.

Jika PowerShell memblokir aktivasi environment, jalankan perintah instalasi dengan interpreter environment secara langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

## W11 - experiment track pada validation

Runner eksperimen penuh ada di `src/w11_experiment.py`. Runner ini memakai protokol CP3 tanpa perubahan (official temporal split, 11.659 evaluable user, candidate universe 22.753 item, relevan rating >= 4, test tidak dipakai) dan menjalankan 20 konfigurasi: 3 baseline, 11 bobot Weighted Score (alpha 0,0-1,0), dan 6 konstanta RRF. Metrik dihitung untuk K = 10, 20, 50 (Recall, Precision, NDCG, MRR, MAP), lalu model terbaik dipilih berdasarkan NDCG@10 dengan tie-breaker Recall@10.

```powershell
python -m src.w11_experiment                 # eksperimen penuh, 11.659 user
python -m src.w11_experiment --limit-users 300 --out outputs/w11_smoke   # smoke test
python -m src.w11_build_report               # PDF + teks analisis RQ
```

Keluaran di `outputs/w11/`: `w11_all_configs.csv` (log semua konfigurasi), `w11_model_comparison.csv`, `w11_user_segments.csv`, `w11_metadata.json` (seed, commit, hash data/cache, audit), `w11_experiment_summary.md`, `w11_rq_analysis.md`, dan `figures/` (PNG 300 dpi + PDF). Laporan hasil eksperimen ada di `output/pdf/W11_Laporan_Hasil_Eksperimen.pdf`.

Ranking komponen dimuat dari cache Task 3 yang sudah diaudit (`outputs/task3/cache`, cache key tercatat di metadata), sehingga metrik mereproduksi run tersebut; W11 tidak melakukan fitting ulang dan tidak menyentuh test set.

## Representasi dan parameter awal

- Interaksi positif ditetapkan sebagai rating `>= 4`.
- Item-CF: matriks biner user-item dari interaksi positif train, cosine similarity item-item, maksimal `top_k=100` tetangga per item.
- TF-IDF CBF: kolom `content_text_basic`, lowercase, stop words English, unigram dan bigram, `min_df=2`, `max_df=0.8`, dan maksimal 100.000 fitur; kemiripan item cosine dengan maksimal 100 tetangga.
- Setiap model membuat ranking kandidat hingga `top_n=200`; item yang sudah dilihat saat train dikecualikan. Skor nol pada Item-CF/CBF diisi memakai urutan popularitas train.
- Metrik utama di notebook adalah macro Recall@10 dan NDCG@10 pada user evaluasi warm. Angka validation adalah hasil pengembangan, bukan hasil test akhir.
- Notebook hybrid memilih bobot Weighted Score dan konstanta RRF berdasarkan validation. Hasil fusi menggabungkan daftar kandidat top-200 dari masing-masing komponen.

## Catatan hasil

Periksa tabel dan contoh rekomendasi yang dihasilkan langsung di notebook. Jangan menafsirkan skor sebagai probabilitas user menyukai item; skor digunakan untuk mengurutkan kandidat. Test set tidak digunakan dalam baseline minggu ini.
