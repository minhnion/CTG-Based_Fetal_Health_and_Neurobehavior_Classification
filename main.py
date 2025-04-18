import os
from utils.data_preprocessing import load_and_preprocess_data

def main():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'data', 'cleaned_data', 'cleaned_data.csv')

    (X_train, y_class_train, y_nsp_train), \
    (X_val,   y_class_val,   y_nsp_val), \
    (X_test,  y_class_test,  y_nsp_test) = load_and_preprocess_data(file_path)

    print(f"Training set: X={X_train.shape}, y_class={y_class_train.shape}, y_nsp={y_nsp_train.shape}")

if __name__ == '__main__':
    main()
