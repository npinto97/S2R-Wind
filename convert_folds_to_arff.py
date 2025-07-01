import os
import pandas as pd
import arff
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
BASE_PATH = "./data/swdpf/T1H12"
OUTPUT_PATH = "./data/swdpf/arff_folds"
TARGET_COL = "patv_target_1"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

for i in range(8):
    fold_path = os.path.join(BASE_PATH, f"fold{i}")
    train_csv = os.path.join(fold_path, "train.csv")
    test_csv = os.path.join(fold_path, "test.csv")

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print(f"Fold {i} missing train/test.")
        continue

    print(f"Processing fold {i}...")

    # Upload and merge
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # === Transform the column "date" ===
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_ts'] = df['date'].astype('int64') // 1_000_000_000
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['dayofweek'] = df['date'].dt.dayofweek
        df = df.drop(columns=["date"])

    # Separate target
    if TARGET_COL in df.columns:
        target = df[TARGET_COL]
        df = df.drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"Target column '{TARGET_COL}' not found in fold {i}.")

    # Keep only numerical input features
    df = df.select_dtypes(include=['number'])

    # === Normalize features ===
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Add back target (not scaled here)
    df_scaled[TARGET_COL] = target.values

    # === Create ARFF ===
    attributes = [(col, 'REAL') for col in df_scaled.columns]
    data = df_scaled.astype(float).values.tolist()

    arff_dict = {
        'relation': f'swdpf_fold{i}',
        'attributes': attributes,
        'data': data
    }

    output_file = os.path.join(OUTPUT_PATH, f'fold{i}.arff')
    with open(output_file, 'w') as f:
        arff.dump(arff_dict, f)

    print(f"Saved: {output_file}")
