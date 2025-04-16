import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ['CLASS', 'NSP', 'SUSP',
                                                              'A','B','C','D','E','AD','DE','LD','FS']]
    X = df[feature_cols].values
    y_class = df['CLASS'].astype(int).values
    y_nsp = df['NSP'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_temp, X_test, y_class_temp, y_class_test, y_nsp_temp, y_nsp_test = train_test_split(
        X_scaled, y_class, y_nsp, test_size=0.1, random_state=42)

    X_train, X_val, y_class_train, y_class_val, y_nsp_train, y_nsp_val = train_test_split(
        X_temp, y_class_temp, y_nsp_temp, test_size=0.1, random_state=42)  

    return (X_train, y_class_train, y_nsp_train), \
           (X_val, y_class_val, y_nsp_val), \
           (X_test, y_class_test, y_nsp_test)
