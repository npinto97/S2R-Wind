import os
import pandas as pd
import arff
import joblib
from sklearn.preprocessing import StandardScaler

BASE_PATH = "./data/swdpf/T1H12"
OUTPUT_PATH = "./data/swdpf/arff_folds"
TARGET_COL = "patv_target_1"

os.makedirs(OUTPUT_PATH, exist_ok=True)

for i in range(8):
    fold_path = os.path.join(BASE_PATH, f"fold{i}")
    train_csv = os.path.join(fold_path, "train.csv")
    test_csv = os.path.join(fold_path, "test.csv")

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print(f"Fold {i} missing train/test.")
        continue

    print(f"Processing fold {i}...")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)

    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_ts'] = df['date'].astype('int64') // 1_000_000_000
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['dayofweek'] = df['date'].dt.dayofweek
        df.drop(columns=["date"], inplace=True)

    # Extract and remove target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in fold {i}.")

    target = df[TARGET_COL].values.reshape(-1, 1)
    df = df.drop(columns=[TARGET_COL])

    # Keep only numerical features
    df = df.select_dtypes(include=['number'])

    # === Normalize input features ===
    scaler_X = StandardScaler()
    df_scaled = pd.DataFrame(scaler_X.fit_transform(df), columns=df.columns)

    # === Normalize target and save both scalers ===
    scaler_y = StandardScaler()
    target_scaled = scaler_y.fit_transform(target).ravel()

    df_scaled[TARGET_COL] = target_scaled

    attributes = [(col, 'REAL') for col in df_scaled.columns]
    data = df_scaled.astype(float).values.tolist()

    arff_dict = {
        'relation': f'swdpf_fold{i}',
        'attributes': attributes,
        'data': data
    }

    arff_path = os.path.join(OUTPUT_PATH, f'fold{i}.arff')
    with open(arff_path, 'w') as f:
        arff.dump(arff_dict, f)
    print(f"Saved ARFF: {arff_path}")

    joblib.dump(scaler_X, os.path.join(OUTPUT_PATH, f'scaler_X_fold{i}.joblib'))
    joblib.dump(scaler_y, os.path.join(OUTPUT_PATH, f'scaler_y_fold{i}.joblib'))
    print(f"Saved scalers for fold {i}")
