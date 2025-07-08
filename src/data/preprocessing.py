import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.helpers import grab_col_names



def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    return df


def get_binary_columns(df, drop_cols=None):
    """
    DataFrame'deki tam olarak 2 sınıfa sahip kategorik sütunları döndürür.
    drop_cols: ['customerID', ...] gibi hariç tutulacak sütunlar
    """
    drop_cols = drop_cols or []
    binary_cols = [
        col for col in df.select_dtypes(include=["object", "category", "bool"]).columns
        if df[col].nunique(dropna=True) == 2 and col not in drop_cols
    ]
    return binary_cols


def get_multi_class_categoricals(df, min_classes=3, drop_cols=None):
    """
    DataFrame'deki ≥ min_classes sınıfa sahip kategorik sütunları döndürür.
    drop_cols: ['customerID', ...] gibi hariç tutulacak sütunlar
    """
    drop_cols = drop_cols or []
    multi_cols = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if df[col].nunique(dropna=True) >= min_classes and col not in drop_cols
    ]
    return multi_cols


def map_binary_columns(df, group_cols):
    """
    Belirtilen sütunlardaki 'Yes/No' ve 'Female/Male' değerlerini 0/1 olarak günceller.
    Yeni sütun oluşturmaz, mevcut sütunları değiştirir.
    """
    bool_map = {"No": 0, "Yes": 1}
    gender_map = {"Female": 0, "Male": 1}

    for col in group_cols:
        if col == "gender":
            df[col] = df[col].map(gender_map)
        elif col == "SeniorCitizen":
            continue  # zaten 0/1, dokunma
        else:
            df[col] = df[col].map(bool_map)

    return df


def feature_engineering(df, cat_cols, num_cols, binary_cols):
    """
    Telco veri seti için özel feature engineering işlemleri:
      - drift_type kategorik değişkeni oluşturma
      - avg_monthly_charge hesaplama
      - is_auto_payment flag'i oluşturma
      - num_services sayısal olarak ekleme
      - ilgili sütunları cat_cols, num_cols, binary_cols listelerine ekleme
    """
    # TotalCharges_ hesapla
    df["TotalCharges_"] = df["MonthlyCharges"] * df["tenure"]

    # drift_type oluştur
    df["charge_drift"] = df["TotalCharges"] - df["TotalCharges_"]
    df["drift_type"] = pd.cut(
        df["charge_drift"],
        bins=[-float("inf"), -100, 100, float("inf")],
        labels=["Yukselmis", "Sabit", "Dusmus"]
    )

    # Geçici sütunları kaldır
    df.drop(columns=["TotalCharges_", "charge_drift"], inplace=True)

    # drift_type → cat_cols
    if "drift_type" not in cat_cols:
        cat_cols.append("drift_type")

    # Ortalama aylık ödeme
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    if "avg_monthly_charge" not in num_cols:
        num_cols.append("avg_monthly_charge")

    # Otomatik ödeme flag
    df["is_auto_payment"] = df["PaymentMethod"].str.lower().str.contains("automatic").astype(int)
    if "is_auto_payment" not in binary_cols:
        binary_cols.append("is_auto_payment")

    # Ek hizmet sayısı (ordinal sayısal olarak bırakılıyor)
    extra_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[extra_services].apply(lambda row: sum(row == "Yes"), axis=1)
    if "num_services" not in num_cols:
        num_cols.append("num_services")

    return df, cat_cols, num_cols, binary_cols


def encode_features(df, cat_cols, num_cols):
    """
    Kategorik ve sayısal değişkenleri encode eder:
      - cat_cols → One-hot encoding (drop_first=True)
      - num_cols → StandardScaler ile ölçekleme
      - 'Churn' varsa encoding dışında tutulur
    """
    # Hedef değişken varsa ayır
    target = None
    if "Churn" in df.columns:
        target = df["Churn"]
        df = df.drop("Churn", axis=1)

    # cat_cols içinde olmayan sütunları filtrele
    valid_cat_cols = [col for col in cat_cols if col in df.columns]

    # One-hot encoding
    df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True, dtype=int)

    # Hedef değişkeni geri ekle (eğitim verisi için)
    if target is not None:
        df["Churn"] = target

    return df


def preprocess_telco_data(df: pd.DataFrame):
    """
    Telco veri seti için tüm preprocessing adımlarını uygular:
      - Eksik değer düzeltme
      - Sütun sınıflandırması (cat, num, cat_but_car)
      - Binary sütunları 0/1'e çevirme
      - Feature engineering
      - Encoding (one-hot + scaling)

    Returns:
        df_encoded: Modellemeye hazır veri seti
        X_cols: Özellik sütunları
        y_col: Hedef sütun adı
    """
    df = df.drop("customerID", axis=1)

    # 1. Eksik değer düzeltme
    df = fix_total_charges(df)

    # 2. Sütun sınıflandırması
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    drop_cols = ["customerID"] + cat_but_car
    binary_cols = get_binary_columns(df, drop_cols=drop_cols)
    cat_cols = [col for col in cat_cols if col not in drop_cols]

    # 3. Binary sütunları 0/1'e çevir
    df = map_binary_columns(df, binary_cols)

    # 4. Feature engineering
    df, cat_cols, num_cols, binary_cols = feature_engineering(df, cat_cols, num_cols, binary_cols)

    # 5. Encoding
    df_encoded = encode_features(df, cat_cols, num_cols)

    # 6. Özellik ve hedef sütunları ayır
    X_cols = [col for col in df_encoded.columns if col != "Churn"]
    y_col = "Churn"

    return df_encoded, X_cols, y_col

