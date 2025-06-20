import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(file_path):
    # 1. Đọc dữ liệu và loại bỏ bản ghi có giá trị thiếu
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    # 2. Chọn các feature cần dùng
    feature_cols = [
        col for col in df.columns
        if col not in ['CLASS', 'NSP', 'SUSP', 'A','B','C','D','E','AD','DE','LD','FS']
    ]
    X = df[feature_cols].values
    y_class = df['CLASS'].astype(int).values - 1
    y_nsp   = df['NSP'].astype(int).values - 1

    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Chia train/test theo tỷ lệ 80% train, 20% test
    X_train, X_test, y_class_train, y_class_test, y_nsp_train, y_nsp_test = train_test_split(
        X_scaled, y_class, y_nsp,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 5. Gói lại thành DataFrame để xuất file CSV
    def create_dataframe(X, y_class, y_nsp):
        df_out = pd.DataFrame(X, columns=feature_cols)
        df_out['CLASS'] = y_class
        df_out['NSP']   = y_nsp
        return df_out

    train_df = create_dataframe(X_train, y_class_train, y_nsp_train)
    test_df  = create_dataframe(X_test,  y_class_test,  y_nsp_test)

    # 6. Lưu ra file
    output_dir = os.path.dirname(file_path)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'),   index=False)

    return (X_train, y_class_train, y_nsp_train), (X_test, y_class_test, y_nsp_test)

def main():
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, 'data', 'cleaned_data', 'cleaned_data.csv')

    (X_train, y_class_train, y_nsp_train), \
    (X_test,  y_class_test,  y_nsp_test) = load_and_preprocess_data(file_path)

    print(f"Training set: X={X_train.shape}, y_class={y_class_train.shape}, y_nsp={y_nsp_train.shape}")
    print(f"   Test set: X={X_test.shape},  y_class={y_class_test.shape},  y_nsp={y_nsp_test.shape}")

if __name__ == '__main__':
    main()
