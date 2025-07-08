from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE, TARGET_COLUMN


def train_test_split_data(df):
    """
    Encode edilmiş veri setini eğitim ve test setlerine ayırır.
    Sayısal değişkenleri ölçekler.

    Parametreler:
        df (pd.DataFrame): Encode edilmiş veri seti
        num_cols (list): Sayısal değişkenlerin listesi

    Dönüş:
        X_train, X_test, y_train, y_test
    """
    # Hedef ve özellik ayrımı
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Train-test bölmesi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
