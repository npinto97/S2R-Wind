import os
import pandas as pd
import numpy as np
import arff
from scipy.io import arff as scipy_arff
from sklearn.utils import shuffle

DATA_DIR = "./data/swdpf/arff_folds"
OUTPUT_DIR = "./data/swdpf/clus_ssl_arffs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALES = [10, 20, 70]
FOLDS = range(8)
TARGET_COL = "patv_target_1"

def load_arff_dataframe(file_path):
    data, meta = scipy_arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df.columns = [col.decode() if isinstance(col, bytes) else col for col in df.columns]
    return df, meta

def save_arff_file(df, meta, file_path):
    attributes = [(col, 'REAL') if col != TARGET_COL else (col, 'REAL') for col in df.columns]
    data = df.replace({np.nan: '?'}, regex=True).values.tolist()
    arff_dict = {
        'description': '',
        'relation': 'ssl_data',
        'attributes': attributes,
        'data': data
    }
    with open(file_path, 'w') as f:
        arff.dump(arff_dict, f)

def make_ssl_arff(df_train, meta, scale, fold):
    df_train = shuffle(df_train, random_state=fold).reset_index(drop=True)
    n_total = len(df_train)
    n_labeled = int(n_total * scale / 100)
    df_ssl = df_train.copy()
    df_ssl.loc[n_labeled:, TARGET_COL] = np.nan
    output_filename = f"fold{fold}_ssl_{scale}.arff"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_arff_file(df_ssl, meta, output_path)
    print(f"Saved: {output_path} | Labeled: {n_labeled} / {n_total}")

def save_test_arff(df_test, meta, fold):
    output_filename = f"fold{fold}_test.arff"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_arff_file(df_test, meta, output_path)
    print(f"Saved test: {output_path}")

for fold in FOLDS:
    # Train
    input_train_path = os.path.join(DATA_DIR, f"fold{fold}_train.arff")
    df_train, meta_train = load_arff_dataframe(input_train_path)
    for scale in SCALES:
        make_ssl_arff(df_train, meta_train, scale, fold)

    # Test
    input_test_path = os.path.join(DATA_DIR, f"fold{fold}_test.arff")
    df_test, meta_test = load_arff_dataframe(input_test_path)
    save_test_arff(df_test, meta_test, fold)
