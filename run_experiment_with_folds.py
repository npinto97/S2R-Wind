import os
import pandas as pd
import yaml
import joblib
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.io import arff

from s2rmslib.create_pairs import generate_file
from s2rmslib.TripleNet.tripleNet import train_triple_net
from s2rmslib.ssrm_optimized import S2RM


# def root_mean_squared_error(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))


def split_labeled_and_unlabeled_data_full(df, scale_percent, seed=0):
    scale_ratio = scale_percent / 100 if scale_percent > 1 else scale_percent
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_labeled = int(len(df) * scale_ratio)
    labeled = df.iloc[:n_labeled]
    unlabeled = df.iloc[n_labeled:]
    return labeled, unlabeled


if __name__ == '__main__':
    with open('configs/config_folds.yml', 'r') as file:
        config = yaml.safe_load(file)

    base_dir = config['experiment_params']['base_data_dir']

    for fold in range(2, 8):  # Set range(8) for the full CV
        train_arff_path = os.path.join(base_dir, f"fold{fold}_train.arff")
        test_arff_path = os.path.join(base_dir, f"fold{fold}_test.arff")
        file_name = f"swdpf_fold{fold}"

        print(f"\n Fold {fold} | Loading train: {train_arff_path}")
        train_data, _ = arff.loadarff(train_arff_path)
        train_df = pd.DataFrame(train_data)

        print(f" Fold {fold} | Loading test: {test_arff_path}")
        test_data, _ = arff.loadarff(test_arff_path)
        test_df = pd.DataFrame(test_data)

        in_channels = train_df.shape[1] - 1
        scaler_y = joblib.load(os.path.join(base_dir, f"scaler_y_fold{fold}.joblib"))

        for scale in config['dataset_params']['scale_labeled_samples']:
            print(f"\n========== Fold {fold} | Scale {scale}% ==========")

            # Split train_df in labeled/unlabeled
            labeled, unlabeled = split_labeled_and_unlabeled_data_full(train_df, scale, seed=fold)

            test = test_df.copy()

            pair_path = os.path.join("training_dataset", str(scale), "pairs_train.csv")
            if os.path.exists(pair_path):
                print(f"[INFO] Skipping pair generation (found at {pair_path})")
            else:
                generate_file(labeled, unlabeled, test, scale)

            model_path = os.path.join("saves", f"{file_name}_best_model.pth")
            if os.path.exists(model_path):
                print(f"[INFO] Skipping TripleNet training (model exists at {model_path})")
            else:
                try:
                    train_triple_net(file_name, in_channels=in_channels, scale=scale)
                    print("[DEBUG] TripleNet training completed successfully")
                except Exception as e:
                    print(f"[ERROR] TripleNet training failed: {e}")

            learners = [
                XGBRegressor(tree_method="hist", device="cuda", n_estimators=100, random_state=1),
                XGBRegressor(tree_method="hist", device="cuda", n_estimators=120, random_state=2),
                XGBRegressor(tree_method="hist", device="cuda", n_estimators=150, random_state=3),
            ]

            reg = S2RM(
                co_learners=learners,
                scale=scale,
                in_channels=in_channels,
                iteration=config['experiment_params']['iteration'],
                tau=config['experiment_params']['tau'],
                gr=config['experiment_params']['gr'],
                K=config['experiment_params']['K']
            )

            reg.fit(file_name, labeled)
            preds = reg.predict(file_name, test, methods=['co_train'])

            # Denormalize predictions and targets
            y_pred = scaler_y.inverse_transform(preds[3].reshape(-1, 1)).ravel()
            y_true = scaler_y.inverse_transform(test.iloc[:, -1].values.reshape(-1, 1)).ravel()

            rmse = root_mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rss = np.sum((y_true - y_pred) ** 2)
            tss = np.sum((y_true - np.mean(y_true)) ** 2)
            rse = rss / tss if tss > 0 else np.nan
            r2 = r2_score(y_true, y_pred)

            print(f"[S2RMS] Fold={fold} | Scale={scale}% | "
                  f"RMSE={rmse:.2f} | MAE={mae:.2f} | RSE={rse:.4f} | R²={r2:.4f}")

            results_dir = f"./results/swdpf_folds/scale_{int(scale)}"
            os.makedirs(results_dir, exist_ok=True)

            pd.DataFrame({
                "fold": [fold],
                "scale": [scale],
                "rmse": [rmse],
                "mae": [mae],
                "rse": [rse],
                "r2": [r2],
                "n_labeled": [len(labeled)],
                "n_unlabeled": [len(unlabeled)],
                "n_train": [len(train_df)],
                "n_test": [len(test_df)]
            }).to_csv(os.path.join(results_dir, f"fold{fold}.csv"), index=False)
