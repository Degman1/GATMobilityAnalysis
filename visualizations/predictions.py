import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_prediction(test_dataloader, y_pred, y_truth, node, node_label, rank, config, num_days=None):
    # Calculate the truth
    s = y_truth.shape
    y_truth = y_truth.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth)

    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, node, 0]
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred)
    
    # Determine the total number of available days
    total_slots = len(y_truth)
    total_days = total_slots // config["SLOTS_PER_DAY"]  # Number of full days available

    # Determine how many days to visualize
    if num_days is None or num_days > total_days:
        num_days = total_days  # Default to all available days

    # Slice the data to the required number of days
    end_idx = num_days * config["SLOTS_PER_DAY"]
    y_truth = y_truth[:end_idx]
    y_pred = y_pred[:end_idx]
    
    t = [t for t in range(0, end_idx)]
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_truth, label="Truth", linestyle="solid", color="red")
    plt.plot(t, y_pred, label="ST-GAT", linestyle="dashed", color="blue")
    plt.xlabel("Time (Hours After Test Data Collection Start Timestamp)")
    plt.ylabel("Population Density")
    plt.title(f"Predictions of Population Density for Node {node_label} Over {num_days} Days")
    plt.legend()
    loc = f"./visualizations/predictions/rank{rank}_node{node_label}_predicted_densities.png"
    plt.savefig(loc)
    plt.close()
    
    print(f"Prediction visualization for node {node_label} saved to {loc}")


def plot_prediction_full(test_dataloader, y_pred, y_truth, node, node_label, rank, config, num_days=None, fine_grained_factor=10):
    # Calculate the truth
    s = y_truth.shape
    y_truth = y_truth.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])
    y_truth = y_truth[:, :, node, 0]  # Extract truth for node
    y_truth = torch.flatten(y_truth)  # Flatten for plotting

    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config["BATCH_SIZE"], config["N_NODE"], s[-1])

    # Extract all 9 prediction steps for the node
    y_preds = [torch.flatten(y_pred[:, :, node, i]) for i in range(s[-1])]

    # Determine the total number of available days
    total_slots = len(y_truth)
    total_days = total_slots // config["SLOTS_PER_DAY"]  # Number of full days available

    # Determine how many days to visualize
    if num_days is None or num_days > total_days:
        num_days = total_days  # Default to all available days

    # Slice the data to the required number of days
    end_idx = num_days * config["SLOTS_PER_DAY"]
    y_truth = y_truth[:end_idx]
    y_preds = [y[:end_idx] for y in y_preds]

    t = np.arange(end_idx)

    # Fine-grained time resolution
    t_fine = np.linspace(0, end_idx - 1, end_idx * fine_grained_factor)  # More points

    plt.figure(figsize=(10, 5))

    # Plot ground truth (interpolated for smoothness)
    truth_interp = interp1d(t, y_truth, kind='cubic')
    plt.plot(t_fine, truth_interp(t_fine), label="Truth", linestyle="solid", color="red")

    # Plot all 9 predictions with proper shifting and smooth curves
    colors = plt.cm.Blues(np.linspace(0.4, 1, 9))  # Varying shades of blue
    for i in range(9):
        shifted_t = t[i:]  # Shift each prediction i steps to the right
        shifted_y = y_preds[i][:len(shifted_t)]  # Ensure matching lengths

        # Interpolate to smooth out curves
        if len(shifted_t) > 1:
            pred_interp = interp1d(shifted_t, shifted_y, kind='cubic', fill_value="extrapolate")
            t_fine_shifted = np.linspace(shifted_t[0], shifted_t[-1], len(shifted_t) * fine_grained_factor)
            plt.plot(t_fine_shifted, pred_interp(t_fine_shifted), linestyle="dashed", color=colors[i], label=f"Pred t+{i+1}")

    plt.xlabel("Time (Hours After Test Data Collection Start Timestamp)")
    plt.ylabel("Population Density")
    plt.title(f"Predictions of Population Density for Node {node_label} Over {num_days} Days")
    plt.legend()
    
    loc = f"./visualizations/predictions/rank{rank}_node{node_label}_predicted_densities.png"
    plt.savefig(loc)
    plt.close()
    
    print(f"Prediction visualization for node {node_label} saved to {loc}")