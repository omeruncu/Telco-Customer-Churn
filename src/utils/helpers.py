import pandas as pd

def data_overview(df: pd.DataFrame) -> None:
    """Veri setinin genel yapısı"""
    print("Veri Seti Genel Bilgisi\n")
    print(f"Gözlem sayısı: {df.shape[0]}")
    print(f"Özellik sayısı: {df.shape[1]}")
    print(f"Sütunlar: {list(df.columns)}\n")

    print("Veri Tipleri ve Bellek Kullanımı:\n")
    df.info()

    print("\nTemel İstatistikler:\n")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("\nEksik (NaN) Değer:\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik (NaN) değer bulunmuyor.")


def grab_col_names(df, cat_th=10, car_th=20, numeric_threshold=0.95):
    """
    Değişkenleri veri tipine ve eşiklere göre sınıflandırır.
    Object tipinde olup büyük oranda sayısal olan sütunları da dönüştürür.

    Parameters
    ----------
    df : pd.DataFrame
    cat_th : int
        Kategorik sayılabilecek değişkenler için sınıf sayısı eşiği
    car_th : int
        Kardinal değişkenler için eşik
    numeric_threshold : float
        Object sütunların sayıya çevrilebilirlik oranı eşiği

    Returns
    -------
    cat_cols : list
        Kategorik değişkenler
    num_cols : list
        Sayısal değişkenler
    cat_but_car : list
        Kategorik görünümlü ama kardinal olan değişkenler
    """

    object_but_numeric = []

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            ratio_numeric = converted.notna().mean()

            if ratio_numeric >= numeric_threshold:
                df[col] = converted
                object_but_numeric.append(col)

    # Güncellenmiş dtype'lara göre sınıflandır
    cat_cols = [col for col in df.columns if df[col].dtype in ["object", "category", "bool"]]
    cat_cols = [col for col in cat_cols if col not in object_but_numeric]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtype in ["int", "float"]]


    cat_but_car = [col for col in cat_cols if df[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Toplam değişken sayısı: {df.shape[1]}")
    print(f"Kategorik değişken sayısı: {len(cat_cols)}")
    print(f"Sayısal değişken sayısı: {len(num_cols)}")
    print(f"Kategorik görünümlü ama kardinal değişken sayısı: {len(cat_but_car)}")

    return cat_cols, num_cols, cat_but_car