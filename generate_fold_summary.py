import pandas as pd
import os
import numpy as np
import re
import csv

S2RMS_RESULTS_PATH = "./results/swdpf_folds"
ELASTICNET_RESULTS_PATH = "./results/elasticnet_folds/elasticnet_fold_results.csv"
XGBOOST_RESULTS_PATH = "./results/xgboost_folds/xgboost_fold_results.csv"
CLUS_OUT_DIR = "./data/swdpf/clus_ssl_settings"
EXCEL_OUTPUT_PATH = "fold_results_summary.xlsx"

SCALES = [10, 20, 70]
NUM_FOLDS = 8


def load_s2rms_results(base_path=S2RMS_RESULTS_PATH, scales=SCALES, num_folds=NUM_FOLDS):
    dfs = []
    for scale in scales:
        for fold in range(num_folds):
            file_path = os.path.join(base_path, f"scale_{scale}", f"fold{fold}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Model"] = "S2RMS"
                df["scale"] = scale
                df["fold"] = fold
                for col in ["r2", "n_train", "n_test"]:
                    if col not in df.columns:
                        df[col] = np.nan
                dfs.append(df)
            else:
                print(f"File not found: {file_path}")
    return pd.concat(dfs, ignore_index=True)


def parse_clus_out_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    match = re.search(r"fold(\d+)_ssl_(\d+)\.out", os.path.basename(filepath))
    if not match:
        return None
    fold = int(match.group(1))
    scale = int(match.group(2))

    mae = rmse = rse = r2 = np.nan
    n_train = n_test = None
    current_section = None

    for i, line in enumerate(lines):
        line = line.strip()

        if "Training error" in line:
            current_section = "Training"
        elif "Testing error" in line:
            current_section = "Testing"

        if line.lower().startswith("number of examples:"):
            try:
                n = int(line.split(":")[1].strip())
                if current_section == "Training":
                    n_train = n
                elif current_section == "Testing":
                    n_test = n
            except:
                continue

        if current_section == "Testing":
            if "Mean absolute error (MAE)" in line:
                for j in range(i+1, i+5):
                    if "Original" in lines[j]:
                        try:
                            mae = float(re.search(r"\[([0-9\.E\-]+)\]", lines[j]).group(1))
                        except:
                            pass
                        break

            if "Root mean squared error (RMSE)" in line:
                for j in range(i+1, i+5):
                    if "Original" in lines[j]:
                        try:
                            rmse = float(re.search(r"\[([0-9\.E\-]+)\]", lines[j]).group(1))
                        except:
                            pass
                        break

            if "Pearson correlation coefficient" in line:
                for j in range(i+1, i+5):
                    if "Original" in lines[j]:
                        try:
                            r2_match = re.search(r"Avg r\^2:\s*([0-9\.E\-]+)", lines[j])
                            if r2_match:
                                r2 = float(r2_match.group(1))
                        except:
                            pass
                        break

    if n_train is not None:
        n_labeled = int(n_train * (scale / 100))
        n_unlabeled = n_train - n_labeled
    else:
        n_labeled = np.nan
        n_unlabeled = np.nan

    # Calcolo RSE leggendo il file .test.pred.arff
    pred_file_path = filepath.replace(".out", ".test.pred.arff")
    y_true = []
    y_pred = []

    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            lines = f.readlines()

        data_started = False
        for line in lines:
            line = line.strip()
            if not data_started:
                if line.lower() == "@data":
                    data_started = True
                continue
            if line == "" or line.startswith("%"):
                continue
            try:
                true_val_str, pred_val_str, _ = next(csv.reader([line]))
                y_true.append(float(true_val_str))
                y_pred.append(float(pred_val_str))
            except Exception as e:
                continue

        if len(y_true) > 1:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            numerator = np.sum((y_true - y_pred) ** 2)
            denominator = np.sum((y_true - np.mean(y_true)) ** 2)
            rse = numerator / denominator if denominator != 0 else np.nan
        else:
            rse = np.nan
    else:
        print(f"[WARNING] Pred file non trovato: {pred_file_path}")
        rse = np.nan

    return {
        "Model": "CLUS+",
        "fold": fold,
        "scale": scale,
        "rmse": rmse,
        "mae": mae,
        "rse": rse,
        "r2": r2,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "n_train": n_train,
        "n_test": n_test
    }


def load_clus_results(out_dir=CLUS_OUT_DIR):
    records = []
    for file in os.listdir(out_dir):
        if file.endswith(".out"):
            path = os.path.join(out_dir, file)
            result = parse_clus_out_file(path)
            if result:
                records.append(result)
    return pd.DataFrame(records)


def prepare_and_merge_data():
    s2rms_df = load_s2rms_results()
    elastic_df = pd.read_csv(ELASTICNET_RESULTS_PATH)
    xgb_df = pd.read_csv(XGBOOST_RESULTS_PATH)
    clus_df = load_clus_results()

    elastic_df["Model"] = "ElasticNet"
    xgb_df["Model"] = "XGBoost"
    elastic_df = elastic_df.rename(columns={"Fold": "fold", "Scale": "scale"})

    for df in [elastic_df, xgb_df]:
        for col in ["n_labeled", "n_unlabeled", "n_train", "n_test", "r2"]:
            if col not in df.columns:
                df[col] = np.nan

    common_cols = ["Model", "fold", "scale", "rmse", "mae", "rse", "r2",
                   "n_labeled", "n_unlabeled", "n_train", "n_test"]

    all_df = pd.concat([
        s2rms_df[common_cols],
        elastic_df[common_cols],
        xgb_df[common_cols],
        clus_df[common_cols]
    ], ignore_index=True)

    return compute_labeled_stats(all_df)


def compute_labeled_stats(all_results_df):
    print("\n[INFO] Example stats per model/scale:")
    stats = all_results_df.groupby(["Model", "scale"])[["n_labeled", "n_unlabeled", "n_train", "n_test"]].mean().round()
    print(stats)
    all_results_df = all_results_df.round(2)
    return all_results_df


def compute_summary(all_results_df):
    summary_df = all_results_df.groupby(["Model", "scale"]).agg(
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        rse_mean=("rse", "mean"),
        r2_mean=("r2", "mean"),
        labeled_avg=("n_labeled", "mean"),
        unlabeled_avg=("n_unlabeled", "mean"),
        train_avg=("n_train", "mean"),
        test_avg=("n_test", "mean")
    ).reset_index()
    summary_df = summary_df.round(2)
    return summary_df


def save_to_excel(all_results_df, summary_df, output_path=EXCEL_OUTPUT_PATH):
    with pd.ExcelWriter(output_path) as writer:
        all_results_df.to_excel(writer, sheet_name="All Folds", index=False)
        summary_df.to_excel(writer, sheet_name="Mean per Scale", index=False)
    print(f"\nExcel file saved to: {output_path}")


if __name__ == "__main__":
    print("Generating Excel report including CLUS+ results...")
    all_df = prepare_and_merge_data()
    summary_df = compute_summary(all_df)
    save_to_excel(all_df, summary_df)
