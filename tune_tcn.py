import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from ray.tune.suggest.hyperopt import HyperOptSearch # Optional: requires hyperopt
import os
import logging
import torch
import numpy as np
import argparse

# Import necessary components from your training script
# Assuming tune_tcn.py is in the same directory or RLTrader is in PYTHONPATH
from tcn_price_predictor.train_tcn import (
    parse_args,
    load_and_prepare_data,
    PriceDataset,
    TCNPricePredictor,
    train_model,
    # evaluate_model is not strictly needed here if train_model returns history
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Logger Setup --- #
# Configure logging for the tune script itself if needed
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tune_trainable(config):
    #\"\"\"
    # Ray Tune trainable function.
    # Takes a hyperparameter configuration, runs one training trial,
    # and reports metrics epoch-by-epoch back to Tune.
    #\"\"\"
    # 1. Get Default Arguments & Override with Tune Config
    default_args = parse_args([])
    args = default_args
    args.tcn_channels = config[\"tcn_channels\"]
    args.tcn_kernel_size = config[\"tcn_kernel_size\"]
    args.tcn_dropout = config[\"tcn_dropout\"]
    args.learning_rate = config[\"learning_rate\"]
    # --- Determine max_epochs based on ASHA scheduler max_t --- #
    # We get max_t from the scheduler config passed by tune.run automatically
    # Defaulting to args.epochs if not running under Tune (e.g., debugging)
    max_epochs = config.get(\"max_t\", args.epochs)
    # Use a fixed, potentially smaller number of epochs for tuning if desired
    # max_epochs = 50 # Example: Fix epochs for tuning


    # 2. Setup (Device, Seeds)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device(\"cuda\")
    else:
        device = torch.device(\"cpu\")

    # 3. Load Data
    try:
        logger.info(f\"Tune Trial: Loading training data from {args.data_path}\")
        X_train, y_train, feature_columns = load_and_prepare_data(
            args.data_path, args.sequence_length, args.target_col, args.prediction_steps
        )
        if X_train.size == 0:
            logger.error(\"Tune Trial: Training data loading resulted in empty sequences.\")
            tune.report(val_dir_acc=float('-inf')) # Report worst score
            return
        train_dataset = PriceDataset(X_train, y_train)
        num_features = X_train.shape[2]

        logger.info(f\"Tune Trial: Loading validation data from {args.val_data_path}\")
        X_val, y_val, _ = load_and_prepare_data(
            args.val_data_path, args.sequence_length, args.target_col, args.prediction_steps
        )
        if X_val.size == 0:
            logger.warning(f\"Tune Trial: Validation data file {args.val_data_path} resulted in zero sequences.\")
            val_dataset = PriceDataset(np.array([]), np.array([]))
        else:
            val_dataset = PriceDataset(X_val, y_val)

    except Exception as e:
        logger.error(f\"Tune Trial: Failed to load data: {e}\", exc_info=True)
        tune.report(val_dir_acc=float('-inf'))
        return

    # Create DataLoaders
    pin_memory = True if device == torch.device(\"cuda\") else False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    # 4. Initialize Model
    model = TCNPricePredictor(
        input_size=num_features,
        output_size=1, 
        num_channels=args.tcn_channels,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout
    ).to(device)

    # 5. Initialize Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    # 6. Initialize Scheduler (Optional)
    lr_scheduler_internal = None
    if args.use_scheduler:
        lr_scheduler_internal = ReduceLROnPlateau(
            optimizer,
            mode=\'min\',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )

    # 7. Training Loop (Replicated and Modified for Tune Reporting)
    epsilon = 1e-10
    logger.info(f\"Tune Trial: Starting training for max {max_epochs} epochs.\")

    for epoch in range(max_epochs):
        # --- Training --- 
        model.train()
        epoch_train_loss = 0.0
        if len(train_loader) > 0:
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_loader)
        else:
            avg_train_loss = float('nan')

        # --- Validation & Metrics --- 
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_preds = []
        epoch_val_targets = []
        avg_val_loss = float('nan')
        val_mae = float('nan')
        val_rmse = float('nan')
        val_dir_acc = float('nan')
        val_custom_acc = float('nan')

        if len(val_loader) > 0:
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    epoch_val_loss += loss.item()
                    epoch_val_preds.extend(outputs.cpu().numpy().flatten())
                    epoch_val_targets.extend(batch_targets.cpu().numpy().flatten())
            
            avg_val_loss = epoch_val_loss / len(val_loader)

            if epoch_val_preds and epoch_val_targets:
                preds_np = np.array(epoch_val_preds)
                targets_np = np.array(epoch_val_targets)
                val_mae = np.mean(np.abs(preds_np - targets_np))
                val_rmse = np.sqrt(np.mean((preds_np - targets_np)**2))
                correct_direction = (np.sign(preds_np) == np.sign(targets_np))
                correct_direction[targets_np == 0] = (preds_np[targets_np == 0] == 0)
                val_dir_acc = np.mean(correct_direction)
                relative_error = np.abs(preds_np - targets_np) / (np.abs(targets_np) + epsilon)
                val_custom_acc = np.mean(relative_error < args.accuracy_threshold)
        
        # --- Internal LR Scheduler Step (based on val_loss) ---
        if lr_scheduler_internal is not None and not np.isnan(avg_val_loss) and np.isfinite(avg_val_loss):
            lr_scheduler_internal.step(avg_val_loss)

        # --- Report Metrics to Ray Tune --- 
        report_dict = {
            \"epoch\": epoch + 1,
            \"val_loss\": avg_val_loss,
            \"val_mae\": val_mae,
            \"val_rmse\": val_rmse,
            \"val_dir_acc\": val_dir_acc,
            \"val_custom_acc\": val_custom_acc,
            \"train_loss\": avg_train_loss,
            \"lr\": optimizer.param_groups[0]['lr']
        }
        # Clean NaNs for reporting, replace with a value Tune can handle (e.g., -inf for accuracy)
        for key, value in report_dict.items():
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                if key == 'val_dir_acc' or key == 'val_custom_acc': # Metrics to maximize
                    report_dict[key] = float('-inf')
                elif key == 'val_loss' or key == 'val_mae' or key == 'val_rmse': # Metrics to minimize
                    report_dict[key] = float('inf')
                else: # Other NaNs (like train_loss if loader empty)
                    report_dict[key] = None # Or keep as NaN if Tune handles it

        tune.report(**report_dict)

        # Note: Early stopping logic based on patience *within* the trial is removed.
        # ASHA scheduler handles stopping trials based on reported metrics across trials.
        # If you still wanted to stop a trial early based on its *own* lack of progress
        # (independent of other trials), you could re-add patience logic here, 
        # but it might interfere with ASHA.

    logger.info(f\"Tune Trial: Finished training after {epoch+1} epochs.\")


if __name__ == \"__main__\":
    # --- Add CLI Arguments for Resources ---
    parser = argparse.ArgumentParser(description=\"Tune TCN Hyperparameters\")
    parser.add_argument(
        \"--cpus_per_trial\",
        type=int,
        default=2,
        help=\"Number of CPUs to allocate per Ray Tune trial.\"
    )
    parser.add_argument(
        \"--gpus_per_trial\",
        type=int,
        default=0,
        help=\"Number of GPUs to allocate per Ray Tune trial.\"
    )
    parser.add_argument(
        \"--num_samples\",
        type=int,
        default=20,
        help=\"Number of hyperparameter combinations to try.\"
    )
    # Add other CLI args if needed (e.g., for local_dir, experiment_name)
    cli_args = parser.parse_args()


    # Define Search Space
    search_space = {
        \"tcn_channels\": tune.choice([
            [64, 128, 256],
            [128, 256, 512],
            [64, 128, 256, 512],
            [128, 256, 512, 512]
        ]),
        # Ensure kernel size is odd and reasonable
        \"tcn_kernel_size\": tune.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
        \"tcn_dropout\": tune.uniform(0.1, 0.6),
        \"learning_rate\": tune.loguniform(1e-5, 1e-3),
        # Add other parameters from 'args' here if you want to tune them, e.g.:
        # \"batch_size\": tune.choice([16, 32, 64]),
    }

    # Configure ASHA Scheduler
    scheduler = ASHAScheduler(
        metric=\"val_dir_acc\", # Metric to compare trials
        mode=\"max\",           # We want to maximize directional accuracy
        # time_attr=\"epoch\", # Use 'epoch' reported in tune.report
        max_t=100,            # Max training epochs per trial (adjust as needed)
        grace_period=10,       # Min epochs before a trial can be stopped
        reduction_factor=2     # Halve the number of trials each round
    )

    # --- Ray Init (Start Ray cluster) ---
    # It's good practice to explicitly initialize Ray
    if not ray.is_initialized():
        ray.init(logging_level=logging.WARNING) # Reduce Ray's internal logging noise

    # --- Ray Tune Execution ---
    analysis = tune.run(
        tune_trainable,
        # Use CLI args for resources
        resources_per_trial={\"cpu\": cli_args.cpus_per_trial, \"gpu\": cli_args.gpus_per_trial},
        config=search_space,
        num_samples=cli_args.num_samples, # Use CLI arg for num_samples
        scheduler=scheduler,
        # search_alg=search_alg, # Uncomment if using HyperOpt
        name=\"tcn_tune_dir_acc_epoch\", # Updated experiment name
        local_dir=\"./ray_results\", # Where to store results
        verbose=1, # 0 = silent, 1 = progress bar, 2 = detailed trial info
        # Add checkpointing config if needed later
        # keep_checkpoints_num=1,
        # checkpoint_score_attr=\"val_dir_acc\",
        # checkpoint_freq=5, # Save every 5 epochs
    )

    # --- Shutdown Ray --- #
    ray.shutdown()

    # Print best result
    best_trial = analysis.get_best_trial(\"val_dir_acc\", \"max\", \"last\")
    if best_trial:
        logger.info(\"--- Best Trial Found ---\")
        # Access the best reported result for the metric
        best_metric_result = analysis.get_best_result(metric=\"val_dir_acc\", mode=\"max\")
        if best_metric_result:
             logger.info(f\"Best Metric (val_dir_acc): {best_metric_result[\"val_dir_acc\"]:.4f} at epoch {best_metric_result[\"epoch\"]}\")
        else:
            logger.warning(\"Could not retrieve best result details.\")
        logger.info(f\"Best Config: {best_trial.config}\")
        logger.info(f\"Log Directory: {best_trial.logdir}\")
    else:
        logger.warning(\"No successful trials completed.\")

    logger.info(\"Ray Tune script finished.\") 