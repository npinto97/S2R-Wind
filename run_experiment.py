import os
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from s2rmslib.create_pairs import generate_file
from s2rmslib.TripleNet.tripleNet import train_triple_net
from s2rmslib.ssrm import S2RM

from scipy.io import arff

def split_labeled_and_unlabeled_data(df, scale, rs):
    training_data = df.sample(2000, random_state=rs)
    labeled_data = training_data.sample(int(2000 * scale), random_state=rs)
    unlabeled_data = training_data.drop(labeled_data.index)
    test_data = df.drop(training_data.index)
    return labeled_data, unlabeled_data, test_data

if __name__ == '__main__':
    with open('configs/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    data_file = "wind.arff"
    file_name = data_file.split('.')[0]

    data, _ = arff.loadarff(os.path.join(config['experiment_params']['base_data_dir'], data_file))
    data = pd.DataFrame(data)

    in_channels = data.shape[1] - 1  # feature number

    for run_iter in range(config['experiment_params']['run_iter']):
        for scale in config['dataset_params']['scale_labeled_samples']:
            print(f"\n========== Run {run_iter} | Scale {scale} ==========")
            labeled, unlabeled, test = split_labeled_and_unlabeled_data(data, scale, run_iter)

            # Prepare file for triplet + co-training
            generate_file(labeled, unlabeled, test, scale)
            train_triple_net(file_name, in_channels=in_channels, scale=scale)

            # Regressors
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

            rmse = mean_squared_error(preds[3], test.iloc[:, -1])
            print(f"[S2RMS] Scale={scale} | Run={run_iter} | RMSE={rmse:.4f}")

            results_dir = f"./results/{file_name}/{int(scale * 100)}"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame({
                "run": [run_iter],
                "scale": [scale],
                "rmse": [rmse]
            }).to_csv(f"{results_dir}/run_{run_iter}.csv", index=False)
