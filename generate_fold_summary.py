import pandas as pd
import os
import numpy as np

def load_s2rms_results(base_path="./results/swdpf_folds", scales=[10, 20, 70], num_folds=1):
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
                print(f"File non trovato: {file_path}")
    return pd.concat(dfs, ignore_index=True)

def prepare_and_merge_data():
    s2rms_df = load_s2rms_results()

    if "n_unlabeled" not in s2rms_df.columns or s2rms_df["n_unlabeled"].isnull().all():
        s2rms_df["n_unlabeled"] = s2rms_df["n_train"] - s2rms_df["n_labeled"]

    elastic_df = pd.read_csv("./results/elasticnet_folds/elasticnet_fold_results.csv")
    elastic_df["Model"] = "ElasticNet"
    elastic_df = elastic_df.rename(columns={"Fold": "fold", "Scale": "scale"})

    xgb_df = pd.read_csv("./results/xgboost_folds/xgboost_fold_results.csv")
    xgb_df["Model"] = "XGBoost"

    for df in [elastic_df, xgb_df]:
        df["n_unlabeled"] = 0

    common_cols = ["Model", "fold", "scale", "rmse", "mae", "rse", "r2",
                   "n_labeled", "n_unlabeled", "n_train", "n_test"]

    for col in ["n_labeled", "n_unlabeled", "n_train", "n_test", "r2"]:
        for df in [elastic_df, xgb_df]:
            if col not in df.columns:
                df[col] = np.nan

    s2rms_df = s2rms_df[common_cols]
    elastic_df = elastic_df[common_cols]
    xgb_df = xgb_df[common_cols]

    all_results_df = pd.concat([s2rms_df, elastic_df, xgb_df], ignore_index=True)

    return compute_labeled_stats(all_results_df)

def compute_labeled_stats(all_results_df):
    print("\nStatistiche sul numero di esempi etichettati e non etichettati per modello e scala:")
    s2rms_only = all_results_df[all_results_df["Model"] == "S2RMS"]
    labeled_stats = s2rms_only.groupby(["scale", "fold"])[["n_labeled", "n_unlabeled", "n_train", "n_test"]].first().reset_index()
    print(labeled_stats)
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
    return summary_df

def save_to_excel(all_results_df, summary_df, output_path="fold_results_summary.xlsx"):
    with pd.ExcelWriter(output_path) as writer:
        all_results_df.to_excel(writer, sheet_name="All Folds", index=False)
        summary_df.to_excel(writer, sheet_name="Mean per Scale", index=False)
    print(f"File Excel salvato come: {output_path}")

if __name__ == "__main__":
    print("Generazione tabella Excel in corso...")
    all_df = prepare_and_merge_data()
    summary_df = compute_summary(all_df)
    save_to_excel(all_df, summary_df)
