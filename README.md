# Sistem Rekomendasi Musical Instruments

Proyek ini membangun dan membandingkan baseline rekomendasi pada dataset Amazon Reviews 2023 kategori Musical Instruments.

## Isi utama

- `01_eda_training.ipynb` ? eksplorasi data training dan katalog item.
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

Buka `02_baseline_recommenders.ipynb`, lalu pilih **Run All**. Setelah itu buka dan jalankan `03_hybrid_fusion.ipynb` untuk membandingkan strategi hybrid. Notebook mengharapkan folder `src/` dan kedua folder dataset berada di bawah direktori proyek.

Jika PowerShell memblokir aktivasi environment, jalankan perintah instalasi dengan interpreter environment secara langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

## Representasi dan parameter awal

- Interaksi positif ditetapkan sebagai rating `>= 4`.
- Item-CF: matriks biner user-item dari interaksi positif train, cosine similarity item-item, maksimal `top_k=100` tetangga per item.
- TF-IDF CBF: kolom `content_text_basic`, lowercase, stop words English, unigram dan bigram, `min_df=2`, `max_df=0.8`, dan maksimal 100.000 fitur; kemiripan item cosine dengan maksimal 100 tetangga.
- Setiap model membuat ranking kandidat hingga `top_n=200`; item yang sudah dilihat saat train dikecualikan. Skor nol pada Item-CF/CBF diisi memakai urutan popularitas train.
- Metrik utama di notebook adalah macro Recall@10 dan NDCG@10 pada user evaluasi warm. Angka validation adalah hasil pengembangan, bukan hasil test akhir.
- Notebook hybrid memilih bobot Weighted Score dan konstanta RRF berdasarkan validation. Hasil fusi menggabungkan daftar kandidat top-200 dari masing-masing komponen.

## Catatan hasil

Periksa tabel dan contoh rekomendasi yang dihasilkan langsung di notebook. Jangan menafsirkan skor sebagai probabilitas user menyukai item; skor digunakan untuk mengurutkan kandidat. Test set tidak digunakan dalam baseline minggu ini.
