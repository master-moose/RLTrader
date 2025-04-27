import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from ray.tune.suggest.hyperopt import HyperOptSearch # Optional: requires hyperopt
import logging
import torch
import numpy as np
import argparse # Need argparse here now
import os # Added: import os

# Import necessary components from your training script
# Assuming tune_tcn.py is in the same directory or RLTrader is in PYTHONPATH
from tcn_price_predictor.train_tcn import (
    # Removed: parse_args,
    add_tcn_training_args, # Import the new function
    load_and_prepare_data,
    PriceDataset,
    TCNPricePredictor,
    # train_model, # Was unused
    # evaluate_model is not strictly needed here if train_model returns history
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Logger Setup --- #
# Configure logging for the tune script itself if needed
# Wrap long line
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Modified signature
def tune_trainable(config, cli_args):
    #\"\"\" # Fixed comment starter
    # Ray Tune trainable function.
    # Takes a hyperparameter configuration and fixed CLI args,
    # runs one training trial, and reports metrics back to Tune.
    #\"\"\" # Fixed comment starter

    # 1. Get Default Arguments & Override with Tune Config
    # Removed: default_args = parse_args([])
    # Removed: args = default_args
    # Use cli_args passed via tune.with_parameters for non-tuned args
    # Use config for tuned hyperparameters
    args_tcn_channels = config["tcn_channels"]
    args_tcn_kernel_size = config["tcn_kernel_size"]
    args_tcn_dropout = config["tcn_dropout"]
    args_learning_rate = config["learning_rate"]

    # --- Determine max_epochs based on ASHA scheduler max_t --- #
    # We get max_t from the scheduler config passed by tune.run automatically
    # Defaulting to cli_args.epochs if not running under Tune
    max_epochs = config.get("max_t", cli_args.epochs)

    # 2. Setup (Device, Seeds) - Use cli_args
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available() and not cli_args.no_cuda:
        torch.cuda.manual_seed(cli_args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. Load Data - Use cli_args
    try:
        # Wrapped long line
        logger.info(
            f"Tune Trial: Loading training data from {cli_args.data_path}"
        )
        # Wrapped long line
        X_train, y_train, feature_columns = load_and_prepare_data(
            cli_args.data_path, cli_args.sequence_length,
            cli_args.target_col, cli_args.prediction_steps
        )
        if X_train.size == 0:
            # Wrapped long line
            logger.error(
                "Tune Trial: Training data loading resulted in empty sequences."
            )
            # Fixed spacing
            tune.report(val_dir_acc=float('-inf'))  # Report worst score
            return
        train_dataset = PriceDataset(X_train, y_train)
        num_features = X_train.shape[2]

        # Wrapped long line
        logger.info(
            f"Tune Trial: Loading validation data from {cli_args.val_data_path}"
        )
        # Wrapped long line
        X_val, y_val, _ = load_and_prepare_data(
            cli_args.val_data_path, cli_args.sequence_length,
            cli_args.target_col, cli_args.prediction_steps
        )
        if X_val.size == 0:
            # Wrapped long line
            logger.warning(
                f"Tune Trial: Validation data file {cli_args.val_data_path} "
                f"resulted in zero sequences."
            )
            val_dataset = PriceDataset(np.array([]), np.array([]))
        else:
            val_dataset = PriceDataset(X_val, y_val)

    except Exception as e:
        logger.error(f"Tune Trial: Failed to load data: {e}", exc_info=True)
        # Correctly report a dictionary of metrics
        tune.report({"val_dir_acc": float('-inf')}) # Report worst score for the metric being optimized
        return

    # Create DataLoaders - Use cli_args
    pin_memory = True if device == torch.device("cuda") else False
    # Wrapped long line
    train_loader = DataLoader(
        train_dataset, batch_size=cli_args.batch_size, shuffle=True,
        num_workers=cli_args.num_workers, pin_memory=pin_memory
    )
    # Wrapped long line
    val_loader = DataLoader(
        val_dataset, batch_size=cli_args.batch_size, shuffle=False,
        num_workers=cli_args.num_workers, pin_memory=pin_memory
    )

    # 4. Initialize Model - Use config for tuned params
    model = TCNPricePredictor(
        input_size=num_features,
        output_size=1, # Assuming output size is always 1 for price prediction
        num_channels=args_tcn_channels, # Use variable from config
        kernel_size=args_tcn_kernel_size, # Use variable from config
        dropout=args_tcn_dropout # Use variable from config
    ).to(device)

    # 5. Initialize Optimizer and Loss - Use config for LR
    optimizer = torch.optim.Adam(model.parameters(), lr=args_learning_rate) # Use variable from config
    criterion = torch.nn.MSELoss()

    # 6. Initialize Scheduler (Optional) - Use cli_args
    lr_scheduler_internal = None
    if cli_args.use_scheduler:
        lr_scheduler_internal = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cli_args.scheduler_factor,
            patience=cli_args.scheduler_patience,
            min_lr=cli_args.scheduler_min_lr,
        )

    # 7. Training Loop (Replicated and Modified for Tune Reporting)
    epsilon = 1e-10
    logger.info(f"Tune Trial: Starting training for max {max_epochs} epochs.")

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        epoch_train_loss = 0.0
        if len(train_loader) > 0:
            for batch_features, batch_targets in train_loader:
                # Wrapped long line
                batch_features, batch_targets = (
                    batch_features.to(device), batch_targets.to(device)
                )
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                # Wrapped long line
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                # Use detach().item() to silence warning
                epoch_train_loss += loss.detach().item()
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
                    # Wrapped long line
                    batch_features, batch_targets = (
                        batch_features.to(device), batch_targets.to(device)
                    )
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    epoch_val_loss += loss.item()
                    # Wrapped long line
                    epoch_val_preds.extend(
                        outputs.cpu().numpy().flatten()
                    )
                    # Wrapped long line
                    epoch_val_targets.extend(
                        batch_targets.cpu().numpy().flatten()
                    )

            avg_val_loss = epoch_val_loss / len(val_loader)

            if epoch_val_preds and epoch_val_targets:
                preds_np = np.array(epoch_val_preds)
                targets_np = np.array(epoch_val_targets)
                val_mae = np.mean(np.abs(preds_np - targets_np))
                val_rmse = np.sqrt(np.mean((preds_np - targets_np)**2))
                correct_direction = (np.sign(preds_np) == np.sign(targets_np))
                # Wrapped long line
                correct_direction[targets_np == 0] = (
                    preds_np[targets_np == 0] == 0
                )
                val_dir_acc = np.mean(correct_direction)
                # Wrapped long line
                relative_error = np.abs(preds_np - targets_np) / (
                    np.abs(targets_np) + epsilon
                )
                # Wrapped long line
                val_custom_acc = np.mean(
                    relative_error < cli_args.accuracy_threshold
                )

        # --- Internal LR Scheduler Step (based on val_loss) ---
        # Wrapped long line
        if (lr_scheduler_internal is not None and
                not np.isnan(avg_val_loss) and
                np.isfinite(avg_val_loss)):
            lr_scheduler_internal.step(avg_val_loss)

        # --- Report Metrics to Ray Tune ---
        report_dict = {
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_dir_acc": val_dir_acc,
            "val_custom_acc": val_custom_acc,
            "train_loss": avg_train_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        # Wrapped long line
        # Clean NaNs for reporting, replace with a value Tune can handle
        for key, value in report_dict.items():
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                # Metrics to maximize; Fixed spacing
                if key == 'val_dir_acc' or key == 'val_custom_acc':
                    report_dict[key] = float('-inf')
                # Metrics to minimize; Fixed spacing
                elif key == 'val_loss' or key == 'val_mae' or key == 'val_rmse':
                    report_dict[key] = float('inf')
                else:  # Other NaNs (like train_loss if loader empty); Fixed spacing
                    report_dict[key] = None  # Or keep as NaN if Tune handles it

        # Pass the dictionary directly instead of unpacking
        # tune.report(**report_dict)
        tune.report(report_dict)

        # Note: Early stopping logic removed, handled by ASHA.
        # Wrapped comments
        # Note: Early stopping logic based on patience *within* the trial is
        # removed. ASHA scheduler handles stopping trials based on reported
        # metrics across trials. If you still wanted to stop a trial early
        # based on its *own* lack of progress (independent of other trials),
        # you could re-add patience logic here, but it might interfere with
        # ASHA.

    logger.info(f"Tune Trial: Finished training after {epoch+1} epochs.")


if __name__ == "__main__":
    # --- Create parser, add train args, add tune args, then parse ---
    # 1. Create a base parser
    parser = argparse.ArgumentParser(description="Tune TCN Hyperparameters using Ray Tune")

    # 2. Add the arguments from train_tcn.py using the helper function
    add_tcn_training_args(parser)

    # 3. Add the tuning-specific arguments to the *same* parser
    parser.add_argument(
        "--cpus_per_trial",
        type=int,
        default=2,
        help="Number of CPUs to allocate per Ray Tune trial."
    )
    parser.add_argument(
        "--gpus_per_trial",
        type=float, # Allow fractional GPUs
        default=0,
        help="Number of GPUs to allocate per Ray Tune trial (can be fractional)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of hyperparameter combinations to try."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="tcn_tune_dir_acc_epoch",
        help="Name for the Ray Tune experiment."
    )

    # 4. Now parse all arguments together
    cli_args = parser.parse_args()

    # --- Construct Storage Path from Experiment Name ---
    # Use a base directory (e.g., current dir or a specific results dir)
    # We use relative path here, assuming script is run from project root
    base_storage_dir = "./ray_results"
    relative_storage_path = os.path.join(base_storage_dir, cli_args.experiment_name)
    # Ensure the path passed to tune.run is absolute
    storage_path = os.path.abspath(relative_storage_path)
    logger.info(f"Ray Tune results will be stored in: {storage_path}")

    # Define Search Space
    search_space = {
        "tcn_channels": tune.choice([
            [64, 128, 256],
            [128, 256, 512],
            [64, 128, 256, 512],
            [128, 256, 512, 512]
        ]),
        # Wrapped long line
        "tcn_kernel_size": tune.choice(
            [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        ),
        "tcn_dropout": tune.uniform(0.1, 0.6),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        # Example: Add batch_size tuning
        # "batch_size": tune.choice([32, 64, 128]),
        # IMPORTANT: If tuning batch_size, make sure tune_trainable uses
        # config["batch_size"] instead of cli_args.batch_size for DataLoaders
    }

    # Configure ASHA Scheduler
    scheduler = ASHAScheduler(
        metric="val_dir_acc",  # Metric to compare trials; Fixed spacing
        mode="max",            # We want to maximize directional accuracy
        # time_attr="epoch",   # Use 'epoch' reported in tune.report
        max_t=cli_args.epochs, # Use epochs from CLI args as max_t
        grace_period=10,       # Min epochs before a trial can be stopped
        reduction_factor=2     # Halve the number of trials each round
    )

    # --- Ray Init (Start Ray cluster) ---
    # It's good practice to explicitly initialize Ray
    if not ray.is_initialized():
        # Wrapped long line; Fixed spacing
        ray.init(
            logging_level=logging.WARNING  # Reduce Ray's internal logging noise
        )

    # --- Ray Tune Execution ---
    analysis = tune.run(
        # Pass cli_args to the trainable function
        tune.with_parameters(tune_trainable, cli_args=cli_args),
        # Use CLI args for resources
        # Wrapped long line
        resources_per_trial={
            "cpu": cli_args.cpus_per_trial, "gpu": cli_args.gpus_per_trial
        },
        config=search_space,
        num_samples=cli_args.num_samples,  # Use CLI arg; Fixed spacing
        scheduler=scheduler,
        # search_alg=search_alg, # Uncomment if using HyperOpt
        name=cli_args.experiment_name,     # Use CLI arg; Fixed spacing
        storage_path=storage_path, # Use constructed path
        verbose=1,  # 0=silent, 1=progress, 2=trial info; Fixed spacing
        # Add checkpointing config if needed later
        # keep_checkpoints_num=1,
        # checkpoint_score_attr="val_dir_acc",
        # checkpoint_freq=5, # Save every 5 epochs
    )

    # --- Shutdown Ray --- #
    ray.shutdown()

    # Print best result
    # Wrapped long line
    best_trial = analysis.get_best_trial(
        "val_dir_acc", "max", "last-5-avg" # More robust selection
    )
    if best_trial:
        logger.info("--- Best Trial Found ---")
        # Access the best reported result dictionary directly from the trial object
        best_metric_result = best_trial.last_result # Get the latest reported metrics from the best trial

        if best_metric_result:
             # Check if the metric key exists before accessing
             metric_key = "val_dir_acc" # The metric we optimized for
             metric_value = best_metric_result.get(metric_key, 'N/A')
             epoch_value = best_metric_result.get('epoch', 'N/A')
             if isinstance(metric_value, float):
                  metric_value_str = f"{metric_value:.4f}"
             else:
                  metric_value_str = str(metric_value)

             logger.info(
                 f"Best Metric ({metric_key}): "
                 f"{metric_value_str} at epoch "
                 f"{epoch_value}"
             )
        else:
             logger.warning("Could not retrieve best result details (last_result) from best_trial object.")
        logger.info(f"Best Config: {best_trial.config}")
        # Get logdir from the trial object itself
        logger.info(
            f"Log Directory: {best_trial.local_path}" # Use local_path instead of logdir
        )
    else:
        logger.warning("No successful trials completed or best trial could not be determined.")

    logger.info("Ray Tune script finished.") 