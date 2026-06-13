"""
automate_Alwan-Fauzi.py

Skrip otomatisasi preprocessing dataset House Prices - Advanced Regression Techniques.
Merupakan konversi dari notebook `Eksperimen_Alwan-Fauzi.ipynb` (bagian Data Preprocessing)
menjadi sebuah fungsi yang dapat dipanggil ulang (reusable) maupun dijalankan via CLI.

Penggunaan CLI:
    python automate_Alwan-Fauzi.py --input house-prices-advanced-regression-techniques_raw/train.csv --output house-prices-advanced-regression-techniques_preprocessing
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Kolom kategorikal dengan NA bermakna "tidak memiliki fitur tersebut" -> diisi "None"
NONE_FILL_COLS = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

# Kolom numerik dengan NA bermakna "tidak ada fitur tersebut" -> diisi 0
ZERO_FILL_COLS = [
    "MasVnrArea", "GarageYrBlt", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageArea", "GarageCars",
]

TARGET_COL = "SalePrice"
ID_COL = "Id"


def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    return pd.read_csv(input_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values sesuai deskripsi pada data_description.txt."""
    df = df.copy()

    for col in NONE_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # Fallback: kolom lain yang masih memiliki NA
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def encode_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """One-hot encoding fitur kategorikal, scaling fitur numerik, dan split train/test."""
    df = df.copy()

    has_target = TARGET_COL in df.columns
    if has_target:
        df[TARGET_COL] = np.log1p(df[TARGET_COL])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if has_target:
        numeric_cols.remove(TARGET_COL)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if has_target:
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        train_out = X_train.copy()
        train_out[TARGET_COL] = y_train.values
        test_out = X_test.copy()
        test_out[TARGET_COL] = y_test.values
        return train_out, test_out

    return df, None


def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline preprocessing lengkap: load -> drop Id -> handle missing values ->
    encode & scale -> split -> simpan ke output_dir.

    Returns: (train_df, test_df)
    """
    df = load_data(input_path)

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    df = handle_missing_values(df)
    train_df, test_df = encode_and_scale(df, test_size=test_size, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)
    if test_df is not None:
        test_df.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset House Prices")
    parser.add_argument("--input", default="house-prices-advanced-regression-techniques_raw/train.csv", help="Path dataset mentah (CSV)")
    parser.add_argument("--output", default="house-prices-advanced-regression-techniques_preprocessing", help="Folder output hasil preprocessing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_df, test_df = preprocess_data(args.input, args.output, args.test_size, args.random_state)
    print(f"Selesai. train: {train_df.shape}, test: {test_df.shape}")
    print(f"Output disimpan di: {args.output}")
