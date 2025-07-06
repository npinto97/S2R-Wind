import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.base import clone
from joblib import Parallel, delayed
from xgboost import DMatrix

from s2rmslib.create_pairs import generate_file
from s2rmslib.select_samples import select_sim_samples
from s2rmslib.transform_data import transform

MAX_CANDIDATES = 500  # Limit retraining to top-N stable samples

class S2RM:
    def __init__(self, co_learners, scale, in_channels, iteration=20, tau=1, gr=1, K=3):
        self.co_learners = co_learners
        self.scale = scale
        self.in_channels = in_channels
        self.iteration = iteration
        self.tau = tau
        self.gr = gr
        self.K = K

    def _init_training_data_and_reg(self, data_name, labeled_data):
        self.labeled_data = labeled_data.reset_index(drop=True)
        self.labeled_l_data = []
        self.labeled_v_data = []
        self.pi = []

        for i, learner in enumerate(self.co_learners):
            inner_data = self.labeled_data.sample(frac=0.8, random_state=42)
            outer_data = self.labeled_data.drop(inner_data.index)

            inner_np = transform(data_name, inner_data, self.in_channels)
            outer_np = transform(data_name, outer_data, self.in_channels)

            self.labeled_l_data.append(inner_np)
            self.labeled_v_data.append(outer_np)
            self.pi.append(None)

            learner.fit(inner_np[:, :-1], inner_np[:, -1])

    def _select_relevant_examples(self, j, unlabeled_data, labeled_v_data, gr, data_name):
        transformed_unlabeled = transform(data_name, unlabeled_data, self.in_channels)
        labeled_X, labeled_y = labeled_v_data[:, :-1], labeled_v_data[:, -1]

        epsilon_j = mean_squared_error(
            self.co_learners[j].predict(labeled_X),
            labeled_y
        )

        other_preds = np.array([
            self.co_learners[i].predict(transformed_unlabeled[:, :-1])
            for i in range(len(self.co_learners)) if i != j
        ])

        std_res = np.std(other_preds, axis=0)
        stable_indices = np.where(std_res < 0.4)[0]

        if len(stable_indices) == 0:
            return np.array([]), []

        # Limit to MAX_CANDIDATES
        if len(stable_indices) > MAX_CANDIDATES:
            stable_indices = stable_indices[:MAX_CANDIDATES]

        mean_preds = np.mean(other_preds[:, stable_indices], axis=0)
        candidate_data = transformed_unlabeled[stable_indices, :-1]
        pred_unlabeled_data = np.hstack([candidate_data, mean_preds.reshape(-1, 1)])

        candidate_df = unlabeled_data.iloc[stable_indices].reset_index(drop=True)
        with_pseudo_label = pd.concat(
            [candidate_df.iloc[:, :-1], pd.DataFrame(mean_preds, columns=[self.labeled_data.columns[-1]])],
            axis=1
        )

        def eval_candidate(x_u):
            augmented_data = np.vstack([self.labeled_l_data[j], x_u])
            model = clone(self.co_learners[j])
            model.fit(augmented_data[:, :-1], augmented_data[:, -1])
            eps_new = mean_squared_error(model.predict(labeled_X), labeled_y)
            return (epsilon_j - eps_new) / (epsilon_j + eps_new)

        delta_scores = Parallel(n_jobs=2, backend="threading")(
            delayed(eval_candidate)(x_u.reshape(1, -1)) for x_u in pred_unlabeled_data
        )

        delta_scores = np.array(delta_scores)
        sorted_indices = np.argsort(delta_scores)[::-1]
        positive_count = min((delta_scores > 0).sum(), gr)

        if positive_count == 0:
            return np.array([]), []

        selected = pred_unlabeled_data[sorted_indices[:positive_count]]
        selected_idx = stable_indices[sorted_indices[:1]]

        self.labeled_data = pd.concat([
            self.labeled_data,
            with_pseudo_label.iloc[sorted_indices[:positive_count]]
        ])

        return selected, selected_idx.tolist()

    def _co_training(self, data_name, selected):
        origin_size = self.labeled_data.shape[0]

        while not selected.empty:
            for i in range(len(self.co_learners)):
                pool = shuffle(selected, random_state=42).reset_index(drop=True)
                selected_pi, idx = self._select_relevant_examples(
                    i, pool, self.labeled_v_data[i], self.gr, data_name
                )
                self.pi[i] = selected_pi
                selected = pool.drop(idx)

            if not any(p.size for p in self.pi):
                break

            for i in range(len(self.co_learners)):
                if self.pi[i].size == 0:
                    continue
                self.labeled_l_data[i] = np.vstack([self.labeled_l_data[i], self.pi[i]])
                self.co_learners[i].fit(
                    self.labeled_l_data[i][:, :-1],
                    self.labeled_l_data[i][:, -1]
                )

        return origin_size, self.labeled_data.shape[0] - origin_size

    def fit(self, data_name, labeled_data):
        self._init_training_data_and_reg(data_name, labeled_data)

        selected, unlabeled_data, is_select_all = select_sim_samples(
            data_name=data_name,
            tau=self.tau,
            in_channels=self.in_channels,
            scale=self.scale
        )

        for it in range(self.iteration):
            if selected.empty:
                break

            prev_size, added = self._co_training(data_name, selected)

            if added <= 3 or is_select_all:
                break

            generate_file(self.labeled_data, unlabeled_data, None, is_first_split=False, scale=self.scale)
            self.tau *= 0.98

            selected, unlabeled_data, is_select_all = select_sim_samples(
                data_name=data_name,
                tau=self.tau,
                in_channels=self.in_channels,
                scale=self.scale,
                data_size=prev_size
            )

    def predict(self, data_name, data, methods=None):
        data_trans = transform(data_name, data, in_channels=self.in_channels)
        data_x = data_trans[:, :-1]
        results = []

        if methods is None:
            methods = ['co_train']

        if 'co_train' in methods:
            preds = []
            for learner in self.co_learners:
                try:
                    x_dmatrix = DMatrix(data_x, device='cuda')
                    y_pred = learner.predict(x_dmatrix)
                except Exception:
                    y_pred = learner.predict(data_x)

                preds.append(y_pred * (1 / len(self.co_learners)))
                results.append(y_pred)

            results.append(np.sum(preds, axis=0))

        return results
