import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
import itertools
import dgl
from dgl.dataloading import GraphDataLoader
import gc
import json
from datetime import datetime

from models.st_gat import ST_GAT
from utils.math import *
import dataloader.breadcrumbs_dataloader
import models.trainer

def load_previous_results():
    # Open the file, load its current content, and add new data
    try:
        with open("results.json", 'r') as file:
            # Load existing data
            return json.load(file)
    except FileNotFoundError:
        # If file doesn't exist, initialize with an empty dictionary
        return {"Results": []}

def train_expanding_window_grid_search(config, param_grid, device, param_index, fold_index):
    """
    Train ST-GAT using an expanding window cross-validation approach with hyperparameter grid search.

    Expanding window schedule:
        Iteration 1: 70% train, 10% val
        Iteration 2: 80% train, 10% val
        Iteration 3: 90% train, 10% val

    :param config: Dictionary containing training configurations.
    :param param_grid: Dictionary containing hyperparameters.
    :param device: Device for training ('cuda' or 'cpu').
    :return: Dictionary with results of each fold, best model state.
    """

    train_ratios = [0.7, 0.8, 0.9]  # Expanding train sizes
    val_ratio = 0.1  # Fixed validation size
    best_model = None
    best_hyperparams = None
    best_val_mae = float('inf')
    results = {}
    
    # Generate all hyperparameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # Individual script changes
    train_ratios = [train_ratios[fold_index]]
    param_combinations = [param_combinations[param_index]]
    
    prev_npred = None
    prev_nhist = None
    
    dataset = None
    num_graphs = None

    for param_set in param_combinations:
        current_params = dict(zip(param_grid.keys(), param_set))
        print(f"\nTesting hyperparameters: {current_params}")
        
        # Rebuild the dataset if N_PRED or N_HIST changes
        if current_params['N_PRED'] != prev_npred or current_params['N_HIST'] != prev_nhist:
            config['N_PRED'] = current_params['N_PRED']
            config['N_HIST'] = current_params['N_HIST']
            prev_npred = current_params['N_PRED']
            prev_nhist = current_params['N_HIST']
            print("N_PRED or N_HIST changed, regenerating graph dataset.")
            dataset, config['D_MEAN'], config['D_STD_DEV'], d_train, d_val, d_test = dataloader.breadcrumbs_dataloader.get_processed_dataset(config)
            config['N_NODE'] = dataset.graphs[0].number_of_nodes()
            num_graphs = len(dataset)
        
        weighted_average_mae = 0

        for fold, train_ratio in enumerate(train_ratios):
            train_size = int(train_ratio * num_graphs)
            val_size = int(val_ratio * num_graphs)

            if train_size + val_size > num_graphs:
                print("ERROR: Train or validation ratios combine to be more data than is present in the dataset")
                break  # Prevent out-of-bounds errors

            train_subset = [dataset[i] for i in range(train_size)]
            val_subset = [dataset[i] for i in range(train_size, train_size + val_size)]

            train_dataloader = dgl.dataloading.GraphDataLoader(train_subset, batch_size=config["BATCH_SIZE"], shuffle=False)
            val_dataloader = dgl.dataloading.GraphDataLoader(val_subset, batch_size=config["BATCH_SIZE"], shuffle=False)

            print(f"Fold {fold}: Train [{len(train_subset)} ({train_ratio})] - Val [{len(val_subset)} ({val_ratio})]")

            # Initialize model with current hyperparameters
            model = ST_GAT(
                in_channels=current_params["N_HIST"],
                out_channels=current_params["N_PRED"],
                n_nodes=config["N_NODE"],
                dropout=current_params["DROPOUT"],
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=current_params["INITIAL_LR"], weight_decay=current_params["WEIGHT_DECAY"])
            loss_fn = torch.nn.MSELoss
            
            if (current_params["INITIAL_LR"] == 1e-4):
                epochs = 100
            elif (current_params["INITIAL_LR"] == 5e-4):
                epochs = 80
            elif (current_params["INITIAL_LR"] == 1e-3):
                epochs = 70
            elif (current_params["INITIAL_LR"] == 5e-3):
                epochs = 50
            else:
                print("ERROR: Epoch cannot be computed.")
                raise
            
            val_mae = None
            
            for epoch in range(epochs):
                train_loss = models.trainer.train(model, device, train_dataloader, optimizer, loss_fn, epoch)

                if epoch % 5 == 0 or epoch == epochs - 1:
                    with torch.no_grad():
                        val_mae, _, _, _, _, _ = models.trainer.eval(model, device, val_dataloader, config, "Valid")
                        val_mae = val_mae.item()
            # Store results for this fold + hyperparameter set            
            print(f"Achieved validation MAE of {val_mae} over {epochs} epochs for fold {fold}")

            # Clear GPU memory
            model.to('cpu')
            del model
            del optimizer
            del train_dataloader
            del val_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
            # Add to weighted validation across folds
            weighted_average_mae += train_ratio * val_mae
            
            # Write the updated data back to the JSON file
            data = load_previous_results()
            with open("results.json", 'w') as file:
                new_data = {
                    "params": current_params,
                    "train_percent": train_ratio,
                    "mae": val_mae,
                    "completion_time": f"{datetime.now()}"
                }
                data["Results"].append(new_data)
                json.dump(data, file, indent=4)
        
        # Average over the weighted mae values
        weighted_average_mae /= len(train_ratios)
        
        # Save best model & hyperparams
        # if weighted_average_mae < best_val_mae:
        #     best_val_mae = weighted_average_mae
        #     # best_model = model.state_dict()
        #     best_hyperparams = current_params
        #     print("NEW BEST PARAMETER SET FOUND!")

    # print("\nBest weighted average validation MAE:", best_val_mae)
    # print("Best hyperparameters:", best_hyperparams)

    return results, best_model, best_hyperparams