# Eksperimen_SML_Alwan-Fauzi

Repository ini berisi eksperimen dataset untuk submission **Membangun Sistem Machine Learning** (Kriteria 1).

## Dataset
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Kaggle) — target: `SalePrice`.

## Struktur
- `Eksperimen_Alwan-Fauzi.ipynb` — notebook eksperimen (data loading, EDA, preprocessing) sesuai Template Eksperimen MSML.
- `automate_Alwan-Fauzi.py` — skrip otomatisasi preprocessing (hasil konversi dari notebook).
- `namadataset_raw/` — dataset mentah (`train.csv`, `data_description.txt`).
- `namadataset_preprocessing/` — dataset hasil preprocessing siap latih (`train_preprocessed.csv`, `test_preprocessed.csv`), dihasilkan otomatis oleh GitHub Actions.
- `.github/workflows/preprocessing.yml` — workflow CI yang menjalankan `automate_Alwan-Fauzi.py` setiap ada perubahan pada dataset mentah / skrip, lalu mem-commit hasilnya.

## Menjalankan secara lokal
```bash
pip install pandas numpy scikit-learn
python automate_Alwan-Fauzi.py --input namadataset_raw/train.csv --output namadataset_preprocessing
```
