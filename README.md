# Hology 2024 - Data Mining Competition Solution

## Deskripsi Proyek
Proyek ini merupakan solusi untuk kompetisi Data Mining Hology 2024, yang berfokus pada klasifikasi gambar fashion (kaos/hoodie dan warna) menggunakan deep learning. Dataset terdiri dari gambar fashion yang harus diprediksi jenis (0=Kaos, 1=Hoodie) dan warnanya (0=merah, 1=kuning, 2=biru, 3=hitam, 4=putih).

## Alur Notebook
Notebook `hology-2024.ipynb` berisi pipeline lengkap mulai dari eksplorasi data, preprocessing, training, hingga inference dan submission:

1. **Persiapan Data**
   - Membaca file CSV label dan sample submission.
   - Mapping file gambar ke ID.
   - Analisis distribusi label dan visualisasi contoh gambar.

2. **Eksplorasi Data (EDA)**
   - Analisis distribusi kelas, crosstab jenis vs warna, dan dimensi gambar.

3. **Preprocessing & Augmentasi**
   - Resize, augmentasi (flip, rotation, color jitter, RandAugment, RandomErasing).
   - Split stratified train/validation.

4. **Modeling**
   - Model multi-task berbasis EfficientNet (dari `timm`), dengan dua output head:
     - Binary classification untuk jenis (BCEWithLogitsLoss)
     - Multi-class classification untuk warna (CrossEntropyLoss)
   - Training dengan AdamW, OneCycleLR/CosineAnnealingLR, dan monitoring Exact Match Ratio (EMR).
   - Advanced: ArcFace, MixUp, CutMix, hyperparameter search.

5. **Ensembling & Semi-Supervised**
   - 5-Fold cross-validation, ensemble model, Test-Time Augmentation (TTA).
   - Pseudo-labeling pada data test confident, retrain dengan data gabungan.

6. **Inference & Submission**
   - Ensemble + TTA untuk prediksi test set.
   - Generate file `submission.csv` siap upload ke leaderboard.

7. **Error Analysis & Iterasi**
   - Analisis error pada validation set.
   - Retrain dengan hyperparameter baru jika diperlukan.

## Cara Menjalankan
1. **Persiapkan lingkungan** (disarankan di Kaggle/Colab dengan GPU):
   - Python 3.8+
   - Library: `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`, `tqdm`
2. **Struktur data** mengikuti format Kaggle:
   - `train.csv`, `sample_submission.csv`/`submission.csv`
   - Folder gambar: `train/train/`, `test/test/`
3. **Jalankan notebook** `hology-2024.ipynb` secara berurutan.
4. **Output**: File `submission.csv`/`submission_ensemble.csv`/`submission_final.csv` akan dihasilkan untuk di-upload.

## Catatan Penting
- Model utama: EfficientNet-B3 (bisa diganti backbone lain dari `timm`).
- Pipeline mendukung advanced augmentasi, ensembling, dan semi-supervised.
- Untuk hasil optimal, gunakan GPU dan lakukan tuning hyperparameter.

## Kontak
Solusi ini dikembangkan untuk kompetisi Hology 2024. Untuk pertanyaan, silakan hubungi tim pengembang. 