import pandas as pd
import os
import numpy as np

def load_s2rms_results(base_path="./results/swdpf_folds", scales=[10, 20, 30], num_folds=8):
    dfs = []
    for scale in scales:
        for fold in range(num_folds):
            file_path = os.path.join(base_path, f"scale_{scale}", f"fold{fold}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Model"] = "S2RMS"
                df["scale"] = scale
                df["fold"] = fold
                df["r2"] = np.nan
                dfs.append(df)
            else:
                print(f"File non trovato: {file_path}")
    return pd.concat(dfs, ignore_index=True)

def prepare_and_merge_data():
    s2rms_df = load_s2rms_results()

    elastic_df = pd.read_csv("./results/elasticnet_folds/elasticnet_fold_results.csv")
    elastic_df["Model"] = "ElasticNet"

    xgb_df = pd.read_csv("./results/xgboost_folds/xgboost_fold_results.csv")
    xgb_df["Model"] = "XGBoost"

    common_cols = ["Model", "fold", "scale", "rmse", "mae", "rse", "r2"]
    elastic_df = elastic_df.rename(columns={"Fold": "fold", "Scale": "scale"})[common_cols]
    xgb_df = xgb_df[common_cols]
    s2rms_df = s2rms_df[common_cols]

    all_results_df = pd.concat([s2rms_df, elastic_df, xgb_df], ignore_index=True)

    return all_results_df

def compute_summary(all_results_df):
    summary_df = all_results_df.groupby(["Model", "scale"]).agg(
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        rse_mean=("rse", "mean"),
        r2_mean=("r2", "mean")
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
