import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, **config):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(config['in_channels'], config['number_of_neurons_1']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config['number_of_neurons_1'], config['number_of_neurons_2'])
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config['number_of_neurons_2'] * 2, 1))

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward_second(self, x):
        return self.fc2(x)

    def forward(self, input1, input2, input3, get_trans_data=False):
        if not get_trans_data:
            output1 = self.forward_once(input1.float())
            output2 = self.forward_once(input2.float())
            output3 = self.forward_once(input3.float())
            concat_ap = th.cat((output1, output2), 2)
            concat_an = th.cat((output1, output3), 2)
            output_p = self.forward_second(concat_ap)
            output_n = self.forward_second(concat_an)
            return output_p.float(), output_n.float()
        else:
            output = self.forward_once(input1.float())
            return output

class PreTripletDataset:
    def __init__(self, training_labeled_path, training_pairs_path):
        self.labeled_data = pd.read_csv(training_labeled_path)
        self.pairs = pd.read_csv(training_pairs_path)

        target_label = self.labeled_data.columns[-1]
        labeled_data = self.labeled_data.drop(columns=[target_label])
        self.features = labeled_data.drop(columns=['index']).to_numpy().astype(np.float32)
        self.indices = labeled_data['index'].to_numpy()

        # Mappa per accesso rapido agli indici
        self.index_to_row = {idx: i for i, idx in enumerate(self.indices)}

        self.triplets = []
        for i, anchor_idx in enumerate(self.indices):
            element_list = self.pairs[
                (self.pairs['Sample1'] == anchor_idx) | (self.pairs['Sample2'] == anchor_idx)]
            if element_list.empty:
                continue
            sorted_element_list = element_list.sort_values(by='Difference', ascending=False)

            pos_idx = sorted_element_list.iloc[0]['Sample2'] if sorted_element_list.iloc[0]['Sample1'] == anchor_idx else sorted_element_list.iloc[0]['Sample1']
            neg_idx = sorted_element_list.iloc[-1]['Sample2'] if sorted_element_list.iloc[-1]['Sample1'] == anchor_idx else sorted_element_list.iloc[-1]['Sample1']

            try:
                anchor_feat = self.features[i]
                positive_feat = self.features[self.index_to_row[pos_idx]]
                negative_feat = self.features[self.index_to_row[neg_idx]]
                self.triplets.append((anchor_feat, positive_feat, negative_feat))
            except KeyError:
                continue

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return (th.tensor(a, dtype=th.float32),
                th.tensor(p, dtype=th.float32),
                th.tensor(n, dtype=th.float32))


class TripletDataset:
    def __init__(self, training_labeled, training_unlabeled, model, number_pos_neg=5, device=None):
        self.labeled_data = pd.read_csv(training_labeled)
        self.unlabeled_data = pd.read_csv(training_unlabeled)
        self.model = model
        self.number_pos_neg = number_pos_neg
        self.device = device if device else ('cuda' if th.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing features
        labeled_features = th.from_numpy(self.labeled_data.iloc[:, 1:-1].to_numpy().astype(np.float32)).to(self.device)
        unlabeled_features = th.from_numpy(self.unlabeled_data.iloc[:, 1:-1].to_numpy().astype(np.float32)).to(self.device)

        with th.no_grad():
            labeled_latent = self.model.forward_once(labeled_features)
            unlabeled_latent = self.model.forward_once(unlabeled_features)

        # Calcolo distanze pairwise tra etichettati e non etichettati
        diff = labeled_latent.unsqueeze(1) - unlabeled_latent.unsqueeze(0)  # N x M x D
        distances = th.norm(diff, dim=2)  # N x M

        # Indici positivi (min distanza) e negativi (max distanza)
        self.pos_indices = distances.topk(self.number_pos_neg, largest=False).indices.cpu().numpy()
        self.neg_indices = distances.topk(self.number_pos_neg, largest=True).indices.cpu().numpy()

        self.labeled_features_np = labeled_features.cpu().numpy()
        self.unlabeled_features_np = unlabeled_features.cpu().numpy()

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        anchor = np.tile(self.labeled_features_np[idx], (self.number_pos_neg, 1))
        positive = self.unlabeled_features_np[self.pos_indices[idx]]
        negative = self.unlabeled_features_np[self.neg_indices[idx]]

        return (th.tensor(anchor, dtype=th.float32),
                th.tensor(positive, dtype=th.float32),
                th.tensor(negative, dtype=th.float32))
