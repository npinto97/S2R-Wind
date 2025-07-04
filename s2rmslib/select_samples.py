import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import yaml

from s2rmslib.TripleNet.model import SiameseNetwork

np.set_printoptions(precision=3, suppress=True)


def load_data(data_name, in_channels, scale):
    print(f"[INFO] Loading config and data for scale={scale} and in_channels={in_channels}...")
    with open('./configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"[ERROR] Failed to load YAML config: {exc}")
            raise

    # Load unlabeled data
    unlabeled_path = os.path.join(config['dataset_params']['dataset_path'], str(scale), 'unlabeled_data.csv')
    print(f"[INFO] Loading unlabeled data from {unlabeled_path}")
    unlabeled_data = pd.read_csv(unlabeled_path)
    unlabeled_data = unlabeled_data.drop(columns=[unlabeled_data.columns[0]])
    unlabeled_data_copy = unlabeled_data.drop(columns=[unlabeled_data.columns[-1]])
    print(f"[INFO] Loaded {len(unlabeled_data)} unlabeled samples.")

    # Load model
    load_path = os.path.join(config['logging_params']['save_dir'], f'{data_name}_best_model.pth')
    print(f"[INFO] Loading model from {load_path}")
    config['model_params']['in_channels'] = int(in_channels)
    model = SiameseNetwork(**config['model_params'])
    model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
    model.eval()
    print(f"[INFO] Model loaded and set to evaluation mode.")

    # Load labeled data
    labeled_path = os.path.join(config['dataset_params']['dataset_path'], str(scale), 'labeled_data.csv')
    print(f"[INFO] Loading labeled data from {labeled_path}")
    labeled_data = pd.read_csv(labeled_path)
    labeled_data = labeled_data.drop(columns=[labeled_data.columns[0], labeled_data.columns[-1]])
    print(f"[INFO] Loaded {len(labeled_data)} labeled samples.")

    return model, labeled_data, unlabeled_data, unlabeled_data_copy


def select_sim_samples(data_name, tau, in_channels, scale, data_size=0):
    print(f"[INFO] Starting selection of similar samples (tau={tau})...")
    model, labeled_data, unlabeled_data, unlabeled_data_copy = load_data(data_name, in_channels, scale)

    has_selected_unlabeled_list = []
    final_selected_list = pd.DataFrame()
    select_all = False

    total_matches = 0
    iteration = 0

    # Convert unlabeled once to tensor for speed
    unlabeled_tensor = th.tensor(unlabeled_data_copy.values, dtype=th.float32)
    seq_len = unlabeled_tensor.shape[1] // in_channels
    unlabeled_tensor = unlabeled_tensor.view(-1, seq_len, in_channels)

    for i in range(len(labeled_data)):
        iteration += 1

        input_tensor_1d = th.tensor(labeled_data.iloc[i].values, dtype=th.float32)
        input_tensor_1 = input_tensor_1d.view(1, -1, in_channels)

        list_similarity = []

        if unlabeled_data.empty:
            print(f"[INFO] Unlabeled data exhausted after {iteration} iterations.")
            break

        with th.no_grad():
            # Repeat labeled sample for batch comparison
            repeated_labeled = input_tensor_1.repeat(unlabeled_tensor.shape[0], 1, 1)
            output, _ = model(repeated_labeled, unlabeled_tensor, unlabeled_tensor)
            list_similarity = output.cpu().numpy().flatten().tolist()

        # Trova i più simili (più basso = più simile)
        selected_index = np.argsort(list_similarity)[:1]
        sims_tau = [list_similarity[int(idx)] for idx in selected_index]
        satisfy_tau = [t for t in sims_tau if t < tau]

        if satisfy_tau:
            print(f"[DEBUG] Match found at iteration {iteration} with similarity {satisfy_tau}")
            selected_index = selected_index[:len(satisfy_tau)]

            has_selected_unlabeled_list.append(unlabeled_data.iloc[selected_index])
            unlabeled_data = unlabeled_data.drop(index=selected_index).reset_index(drop=True)
            unlabeled_data_copy = unlabeled_data_copy.drop(index=selected_index).reset_index(drop=True)

            # Ricalcola il tensore ridotto
            unlabeled_tensor = th.tensor(unlabeled_data_copy.values, dtype=th.float32)
            if unlabeled_tensor.shape[0] > 0:
                unlabeled_tensor = unlabeled_tensor.view(-1, seq_len, in_channels)

            total_matches += len(satisfy_tau)

    if has_selected_unlabeled_list:
        final_selected_list = pd.concat(has_selected_unlabeled_list)
        print(f"[INFO] Total selected samples: {len(final_selected_list)}")
    else:
        print("[INFO] No samples selected based on similarity threshold.")

    if unlabeled_data.empty:
        print("[INFO] All unlabeled samples have been used.")
        select_all = True
    else:
        print(f"[INFO] Remaining unlabeled samples: {len(unlabeled_data)}")

    return final_selected_list, unlabeled_data, select_all
