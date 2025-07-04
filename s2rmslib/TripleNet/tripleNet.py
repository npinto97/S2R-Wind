import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from s2rmslib.TripleNet.model import *
from s2rmslib.TripleNet.training import *


def get_pre_data_loaders(**config):
    print("[INFO] Loading pre-triplet dataset...")
    pre_triplet_dataset = PreTripletDataset(
        os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
        os.path.join(config['dataset_params']['dataset_path'], 'pairs_train.csv')
    )
    pre_triplet_loader = DataLoader(
        pre_triplet_dataset, 
        shuffle=True, 
        num_workers=8,
        batch_size=config['exp_params']['batch_size']
    )
    print("[INFO] Pre-triplet dataset loaded successfully.")
    return pre_triplet_loader


def get_cur_data_loaders(model, number_pos_neg, **config):
    print("[INFO] Loading current triplet dataset...")
    triplet_dataset = TripletDataset(
        os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
        os.path.join(config['dataset_params']['dataset_path'], 'unlabeled_data.csv'),
        model,
        number_pos_neg
    )
    triplet_loader = DataLoader(
        triplet_dataset, 
        shuffle=True, 
        num_workers=0,
        batch_size=config['exp_params']['batch_size']
    )
    print("[INFO] Current triplet dataset loaded successfully.")
    return triplet_loader


def train(epochs, data_loader, model, optimizer, device, in_channels, data='', save_path=None):
    print(f"[INFO] Starting training for {epochs} epochs...")
    list_train_loss = []
    cur_min_val_error = float(1e8)

    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs}...")
        model.train()
        train_loss = train_triplet(data_loader, model, optimizer, device, in_channels)
        list_train_loss.append(train_loss)

        print(f"[LOG] Epoch {epoch+1} - Loss: {train_loss:.4f}")

        if train_loss < cur_min_val_error:
            if save_path:
                model_path = os.path.join(save_path, f'{data}_best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"[INFO] New best model saved: {model_path}")
            cur_min_val_error = train_loss

    print("[INFO] Training completed.")


def train_triple_net(data_name, in_channels, scale):
    print("[INFO] Loading configuration from config.yml...")
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
            print("[INFO] Configuration loaded successfully.")
        except yaml.YAMLError as exc:
            print("[ERROR] Failed to parse YAML file:", exc)
            return

    config['model_params']['in_channels'] = int(in_channels)
    config['dataset_params']['dataset_path'] = os.path.join(config['dataset_params']['dataset_path_fixed'], str(scale))
    config['dataset_params']['evaluation_path'] = os.path.join(config['dataset_params']['evaluation_path_fixed'], str(scale))

    save_path = os.path.join(config['logging_params']['save_dir'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"[INFO] Save directory created at: {save_path}")

    print("[INFO] Setting random seeds for reproducibility...")
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    print("[INFO] Initializing model...")
    model = SiameseNetwork(**config['model_params']).to(device)

    optimizer_name = config['exp_params']['optimizer']
    print(f"[INFO] Selected optimizer: {optimizer_name}")
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['exp_params']['LR'], momentum=0.9)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['exp_params']['LR'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['exp_params']['LR'])

    print("[INFO] Preparing pre-triplet data loader...")
    pre_triplet_loader = get_pre_data_loaders(**config)

    print("[INFO] Starting training on pre-triplet dataset...")
    train(config['exp_params']['epochs'], pre_triplet_loader, model, optimizer, device,
          config['model_params']['in_channels'], data_name, save_path)

    # If you want to enable the second training stage, uncomment the lines below.
    # print("[INFO] Starting training on current triplet dataset...")
    # triplet = SiameseNetwork(**config['model_params']).to(device)
    # optimizer = torch.optim.Adam(triplet.parameters(), lr=config['exp_params']['LR'])
    # triplet_loader = get_cur_data_loaders(model, config['model_params']['number_pos_neg'], **config)
    # train(config['exp_params']['epochs'], triplet_loader, triplet, optimizer, device,
    #       config['model_params']['in_channels'], data_name, save_path)
