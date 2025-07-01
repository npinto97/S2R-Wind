import os
import pandas as pd
import yaml
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.io import arff
from sklearn.metrics import mean_absolute_error


from s2rmslib.create_pairs import generate_file
from s2rmslib.TripleNet.tripleNet import train_triple_net
from s2rmslib.ssrm import S2RM

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def split_labeled_and_unlabeled_data(df, scale, rs):
    training_data = df.sample(2000, random_state=rs)
    labeled_data = training_data.sample(int(2000 * scale), random_state=rs)
    unlabeled_data = training_data.drop(labeled_data.index)
    test_data = df.drop(training_data.index)
    return labeled_data, unlabeled_data, test_data

if __name__ == '__main__':
    with open('configs/config_folds.yml', 'r') as file:
        config = yaml.safe_load(file)

    data_dir = config['experiment_params']['base_data_dir']
    scaler_y = joblib.load('./models/scaler_y.joblib')

    for fold in range(8):
        data_path = os.path.join(data_dir, f"fold{fold}.arff")
        file_name = f"swdpf_fold{fold}"

        print(f"\nFold {fold} | Loading {data_path}")
        data, _ = arff.loadarff(data_path)
        data = pd.DataFrame(data)

        in_channels = data.shape[1] - 1

        for scale in config['dataset_params']['scale_labeled_samples']:
            print(f"\n========== Fold {fold} | Scale {scale} ==========")

            labeled, unlabeled, test = split_labeled_and_unlabeled_data(data, scale, rs=fold)

            generate_file(labeled, unlabeled, test, scale)
            pair_path = os.path.join("training_dataset", str(scale), "pairs_train.csv")
            if os.path.exists(pair_path):
                df_pairs = pd.read_csv(pair_path)
                print(f"Pair samples: {len(df_pairs)} | Mean diff: {df_pairs['Difference'].mean():.4f}")
            else:
                print("pairs_train.csv not found!")

            train_triple_net(file_name, in_channels=in_channels, scale=scale)

            learners = [
                RandomForestRegressor(n_estimators=100, random_state=1),
                RandomForestRegressor(n_estimators=120, random_state=2),
                RandomForestRegressor(n_estimators=150, random_state=3),
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

            # Compute metrics
            rmse = root_mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            # RSE = RSS / TSS
            rss = np.sum((y_true - y_pred) ** 2)
            tss = np.sum((y_true - np.mean(y_true)) ** 2)
            rse = rss / tss if tss > 0 else np.nan

            print(f"[S2RMS] Fold={fold} | Scale={scale} | RMSE={rmse:.2f} | MAE={mae:.2f} | RSE={rse:.4f}")

            results_dir = f"./results/swdpf_folds/scale_{int(scale * 100)}"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame({
                "fold": [fold],
                "scale": [scale],
                "rmse": [rmse],
                "mae": [mae],
                "rse": [rse]
            }).to_csv(os.path.join(results_dir, f"fold{fold}.csv"), index=False)

