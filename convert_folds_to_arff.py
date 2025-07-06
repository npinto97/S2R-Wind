import os
import pandas as pd
import arff
import joblib
from sklearn.preprocessing import StandardScaler

BASE_PATH = "./data/swdpf/T1H12"
RAW_OUTPUT_PATH = "./data/swdpf/arff_folds"
SCALED_OUTPUT_PATH = "./data/swdpf/arff_scaled_folds"
TARGET_COL = "patv_target_1"

os.makedirs(RAW_OUTPUT_PATH, exist_ok=True)
os.makedirs(SCALED_OUTPUT_PATH, exist_ok=True)

for i in range(8):
    fold_path = os.path.join(BASE_PATH, f"fold{i}")
    train_csv = os.path.join(fold_path, "train.csv")
    test_csv = os.path.join(fold_path, "test.csv")

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print(f"[WARNING] Fold {i} missing train/test.")
        continue

    print(f"\n[INFO] Processing fold {i}...")

    # Load datasets
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Optional datetime processing
    for df in [df_train, df_test]:
        if "date" in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date_ts'] = df['date'].astype('int64') // 1_000_000_000
            df['hour'] = df['date'].dt.hour
            df['minute'] = df['date'].dt.minute
            df['dayofweek'] = df['date'].dt.dayofweek
            df.drop(columns=["date"], inplace=True)

    # Separate target
    if TARGET_COL not in df_train.columns or TARGET_COL not in df_test.columns:
        raise ValueError(f"[ERROR] Target column '{TARGET_COL}' not found in fold {i}.")

    y_train = df_train.pop(TARGET_COL).values.reshape(-1, 1)
    y_test = df_test.pop(TARGET_COL).values.reshape(-1, 1)

    # Keep only numeric features
    df_train = df_train.select_dtypes(include=['number'])
    df_test = df_test.select_dtypes(include=['number'])

    # Add target back for raw version
    df_train_raw = df_train.copy()
    df_test_raw = df_test.copy()
    df_train_raw[TARGET_COL] = y_train.ravel()
    df_test_raw[TARGET_COL] = y_test.ravel()

    def save_arff(df_scaled, split, fold_index, output_dir):
        attributes = [(col, 'REAL') for col in df_scaled.columns]
        data = df_scaled.astype(float).values.tolist()
        arff_dict = {
            'relation': f'swdpf_fold{fold_index}_{split}',
            'attributes': attributes,
            'data': data
        }
        arff_path = os.path.join(output_dir, f'fold{fold_index}_{split}.arff')
        with open(arff_path, 'w') as f:
            arff.dump(arff_dict, f)
        print(f"[INFO] Saved ARFF: {arff_path}")

    # Save raw (non-normalized) ARFF files
    save_arff(df_train_raw, "train", i, RAW_OUTPUT_PATH)
    save_arff(df_test_raw, "test", i, RAW_OUTPUT_PATH)

    # Normalize
    scaler_X = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler_X.fit_transform(df_train), columns=df_train.columns)
    df_test_scaled = pd.DataFrame(scaler_X.transform(df_test), columns=df_test.columns)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    y_test_scaled = scaler_y.transform(y_test).ravel()

    df_train_scaled[TARGET_COL] = y_train_scaled
    df_test_scaled[TARGET_COL] = y_test_scaled

    # Save scaled ARFF files
    save_arff(df_train_scaled, "train", i, SCALED_OUTPUT_PATH)
    save_arff(df_test_scaled, "test", i, SCALED_OUTPUT_PATH)

    # Save scalers
    joblib.dump(scaler_X, os.path.join(SCALED_OUTPUT_PATH, f'scaler_X_fold{i}.joblib'))
    joblib.dump(scaler_y, os.path.join(SCALED_OUTPUT_PATH, f'scaler_y_fold{i}.joblib'))
    print(f"[INFO] Saved scalers for fold {i}")
