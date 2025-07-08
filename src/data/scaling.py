from sklearn.preprocessing import RobustScaler

def scale_numeric_features(X_train, X_test, num_cols):
    scaler = RobustScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test
