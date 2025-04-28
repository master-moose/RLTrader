# generate_synthetic_lob.py
import argparse
import logging
from pathlib import Path
import time
# import h5py # No longer needed directly here if saving logic changes or is handled elsewhere
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE  # Added for t-SNE
from statsmodels.graphics.tsaplots import plot_acf  # Added for ACF
from torch.optim.lr_scheduler import ReduceLROnPlateau # Added

# --- Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Reformat to comply with line length
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --- Diffusion Model Components ---

# Sinusoidal Time Embedding (Helper function)
def sinusoidal_embedding(n_steps, dim):
    out = torch.zeros(n_steps, dim)
    position = torch.arange(0, n_steps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim)
    )
    out[:, 0::2] = torch.sin(position * div_term)
    out[:, 1::2] = torch.cos(position * div_term)
    return out


# Transformer Model for Noise Prediction
class TransformerDiffusion(nn.Module):
    def __init__(
        self,
        n_steps,
        num_features,
        sequence_length,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Time embedding
        self.time_embed = nn.Embedding(n_steps, d_model)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, d_model)
        self.time_embed.requires_grad_(False)

        # Positional encoding for sequence
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Expect (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x, t):
        # x shape: (batch_size, sequence_length, num_features)
        # t shape: (batch_size,)

        # Project input features to d_model
        x_proj = self.input_proj(x)  # (batch, seq, d_model)

        # Get time embedding and expand it to match sequence length
        # (batch_size,) -> (batch_size, d_model)
        t_emb = self.time_embed(t)
        # (batch_size, d_model) -> (batch_size, 1, d_model)
        t_emb = t_emb.unsqueeze(1)
        # (batch_size, 1, d_model) -> (batch_size, seq, d_model)
        t_emb = t_emb.expand(-1, self.sequence_length, -1)

        # Combine input projection, positional encoding, and time embedding
        combined_input = x_proj + self.pos_encoder + t_emb

        # Pass through Transformer
        # Output shape: (batch, seq, d_model)
        transformer_output = self.transformer_encoder(combined_input)

        # Project back to original feature dimension
        # Output shape: (batch, seq, num_features)
        noise_pred = self.output_proj(transformer_output)
        return noise_pred


# Diffusion Scheduler (DDPM) - Modified slightly for sequence input
class DDPMScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device='cuda'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1.0 - self.alpha_cumprod
        )

    def add_noise(self, original_samples, timesteps):
        # original_samples shape: (batch_size, sequence_length, num_features)
        # timesteps shape: (batch_size,)

        # Expand timesteps for broadcasting: (batch_size,) -> (batch, 1, 1)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].unsqueeze(
            -1
        ).unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = \
            self.sqrt_one_minus_alpha_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn_like(original_samples, device=self.device)
        noisy_samples = (
            sqrt_alpha_cumprod_t * original_samples
            + sqrt_one_minus_alpha_cumprod_t * noise
        )
        return noisy_samples, noise

    def step(self, model_output_noise, timestep, sample):
        # Denoise one step using DDPM formula
        # sample shape: (batch_size, sequence_length, num_features)
        # model_output_noise shape: (batch_size, sequence_length, num_features)
        # timestep: scalar integer

        t = timestep
        pred_noise = model_output_noise

        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]
        beta_t = self.betas[t]

        # Calculate predicted original sample (x_0) from noise prediction
        pred_original_sample = (
            sample - sqrt_one_minus_alpha_cumprod_t * pred_noise
        ) / torch.sqrt(alpha_cumprod_t)
        # Clip predicted x_0 (optional, depends on data range)
        # pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # Calculate mean of q(x_{t-1} | x_t, x_0)
        if t == 0:
            # Simplified formula for t=0, directly use pred_original_sample?
            # Or use the formula which simplifies? Let's use the general one.
             prev_sample_mean = (
                (pred_original_sample * torch.sqrt(self.alphas[t])) +
                (sample - self.sqrt_alpha_cumprod[t] * pred_original_sample) *
                torch.sqrt(1.0 - self.alphas[t]) /
                torch.sqrt(1.0 - self.alpha_cumprod[t])
             ) / torch.sqrt(self.alphas[t])

        else:
            alpha_cumprod_prev = self.alpha_cumprod[t - 1]
            # posterior_variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t # Not directly needed for mean

            # Formula for mean coefficient 1
            mean_coef1 = beta_t * torch.sqrt(alpha_cumprod_prev) \
                / (1.0 - alpha_cumprod_t)
            # Formula for mean coefficient 2
            mean_coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha_t) \
                / (1.0 - alpha_cumprod_t)
            prev_sample_mean = (
                mean_coef1 * pred_original_sample + mean_coef2 * sample
            )

        # Calculate variance and noise for the step
        # Use simplified variance (beta_t) or the interpolated one? DDPM uses beta_t.
        # variance = beta_t
        # Let's use the posterior variance calculation from DDPM paper eq. 7
        # posterior_variance_t = beta_t * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod_t)
        # Simplified: Use beta_t for t > 0, 0 for t = 0
        variance = self.betas[t]
        noise = torch.randn_like(sample)

        # Combine mean and noise
        prev_sample = prev_sample_mean + torch.sqrt(variance) * noise if t > 0 \
            else prev_sample_mean # No noise added at the last step (t=0)

        return prev_sample


# --- Data Handling ---
class LOBDataset(Dataset):
    def __init__(self, data_tensor, sequence_length):
        # Expect data_tensor to be the pre-converted tensor
        self.data = data_tensor
        self.sequence_length = sequence_length
        # Calculate number of sequences considering the sequence length
        self.num_sequences = len(data_tensor) - sequence_length + 1
        if self.num_sequences <= 0:
            raise ValueError(
                f"Data length ({len(data_tensor)}) is less than sequence length "
                f"({sequence_length}). Cannot create sequences."
            )

    def __len__(self):
        # Each item will be a sequence
        return self.num_sequences

    def __getitem__(self, idx):
        # Return a sequence (slice) from the pre-existing tensor
        # Shape: (sequence_length, num_features)
        sequence_slice = self.data[idx : idx + self.sequence_length]
        # No need to convert to tensor here, it already is one
        return sequence_slice


class Scaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit_transform(self, data):
        # Simple min-max scaling to [0, 1]
        # Ensure data is numpy array for nanmin/nanmax
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.min_ = np.nanmin(data, axis=0, keepdims=True)
        self.max_ = np.nanmax(data, axis=0, keepdims=True)
        # Handle cases where min == max
        range_ = self.max_ - self.min_
        # Avoid division by zero if a feature is constant
        range_[range_ == 0] = 1.0
        scaled_data = (data - self.min_) / range_
        # Fill NaNs resulting from scaling or original data
        scaled_data = np.nan_to_num(scaled_data, nan=0.5)  # Fill with mid-value
        return scaled_data

    def inverse_transform(self, scaled_data):
        if self.min_ is None or self.max_ is None:
            raise Exception("Scaler not fitted yet.")
        # Ensure data is numpy array
        if isinstance(scaled_data, torch.Tensor):
            scaled_data = scaled_data.cpu().numpy()

        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1.0  # Use the same range handling
        original_data = scaled_data * range_ + self.min_
        return original_data


def load_and_preprocess_data(
    data_dir, symbol_filename_part, hdf_key, lob_depth
):
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob(f"{symbol_filename_part}_lob.h5")) #temporarily use downloaded dataset
    if not all_files:
        raise FileNotFoundError(
            f"No HDF5 files found in {data_dir} matching "
            f"{symbol_filename_part}_lob_*.h5"
        )

    logging.info(f"Loading data from {len(all_files)} HDF5 files...")
    df_list = []
    for file in tqdm(all_files, desc="Loading HDF5 files"):
        try:
            # Only load necessary columns if possible (reduces memory)
            # This requires knowing column names beforehand or reading metadata
            df_list.append(pd.read_hdf(file, key=hdf_key))
        except Exception as e:
            logging.warning(f"Could not read file {file}: {e}")
            continue  # Skip problematic files

    if not df_list:
        raise ValueError("No valid data loaded from HDF5 files.")

    df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Loaded total {len(df)} rows.")

    # Select features (all bid/ask prices and sizes)
    feature_cols = []
    for i in range(lob_depth):
        feature_cols.extend([
            f'bid_price_{i}', f'bid_size_{i}',
            f'ask_price_{i}', f'ask_size_{i}'
        ])

    # Ensure all expected columns exist, handle missing ones if necessary
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected LOB columns in HDF5 data: {missing_cols}"
        )

    # Check for timestamp column
    if 'timestamp' not in df.columns:
        logging.warning(
            "Timestamp column not found. Synthetic timestamps will start "
            "from current time."
        )
        timestamps = pd.Series([], dtype='datetime64[ns, UTC]') # Empty series
    else:
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        # Handle potential errors during conversion
        if timestamps.isnull().any():
             logging.warning("Some timestamps failed to parse. Check data.")
             timestamps = timestamps.fillna(method='ffill').fillna(method='bfill') # Basic imputation


    data = df[feature_cols].values.astype(np.float32)

    # Handle potential NaNs before scaling
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        logging.warning(
            f"NaNs found in data ({nan_count} instances, "
            f"{nan_count / data.size * 100:.2f}%). Filling with column means."
        )
        col_means = np.nanmean(data, axis=0)
        # Handle columns that are all NaN
        col_means = np.nan_to_num(col_means)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_means, inds[1])

    # Normalize data
    scaler = Scaler()
    scaled_data_np = scaler.fit_transform(data)
    logging.info(f"Data shape after scaling: {scaled_data_np.shape}")

    # Convert the entire dataset to a tensor *once*
    scaled_data_tensor = torch.tensor(scaled_data_np, dtype=torch.float32)
    logging.info("Converted scaled data to PyTorch tensor.")

    return scaled_data_tensor, scaler, timestamps, feature_cols


# --- Training Function ---
def train_diffusion_model(
    model, scheduler, dataloader, num_epochs, lr, device
):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Add ReduceLROnPlateau scheduler
    scheduler_lr = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()  # Use MSE to predict the noise
    model.train()
    losses = []

    logging.info("Starting diffusion model training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_sequences in pbar:
            # batch_sequences shape: (batch_size, sequence_length, num_features)
            batch_sequences = batch_sequences.to(device)
            batch_size = batch_sequences.size(0)

            # Sample random timesteps for each sequence in the batch
            # t shape: (batch_size,)
            t = torch.randint(
                0, scheduler.num_timesteps, (batch_size,), device=device
            ).long()

            # Add noise according to DDPM schedule
            # noise shape: (batch_size, sequence_length, num_features)
            noisy_batch, noise = scheduler.add_noise(batch_sequences, t)

            # Predict noise using the Transformer model
            optimizer.zero_grad()
            # predicted_noise shape: (batch_size, sequence_length, num_features)
            predicted_noise = model(noisy_batch, t)

            # Calculate loss between predicted noise and actual noise
            loss = criterion(predicted_noise, noise)

            # Backpropagate
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}"
        )

        # Step the LR scheduler based on the average epoch loss
        scheduler_lr.step(avg_epoch_loss)

    logging.info("Training finished.")
    return losses


# --- Sampling Function ---
@torch.no_grad()
def sample_diffusion_model(
    model, scheduler, num_samples, num_features, sequence_length, device,
    batch_size=64 # Added batch size for sampling efficiency
):
    model.eval()
    logging.info(f"Generating {num_samples} synthetic sequences...")

    all_generated_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Sampling Batches"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        if current_batch_size <= 0:
            break

        # Start with random noise sequences
        # Shape: (current_batch_size, sequence_length, num_features)
        samples = torch.randn(
            current_batch_size, sequence_length, num_features, device=device
        )

        # Denoise step by step from T to 0
        for t in range(scheduler.num_timesteps - 1, -1, -1):
            # Timestep tensor for the batch
            # Shape: (current_batch_size,)
            timestep = torch.tensor(
                [t] * current_batch_size, device=device
            ).long()

            # Predict noise for the current noisy samples
            # Shape: (current_batch_size, sequence_length, num_features)
            predicted_noise = model(samples, timestep)

            # Get the less noisy sample x_{t-1} using the scheduler's step function
            # Shape: (current_batch_size, sequence_length, num_features)
            samples = scheduler.step(predicted_noise, t, samples)

            # Optional: Add clipping or other constraints during sampling if needed
            # samples = torch.clamp(samples, -1.0, 1.0) # If data was scaled to [-1, 1]

        all_generated_sequences.append(samples.cpu())

    # Concatenate sequences from all batches
    generated_sequences_tensor = torch.cat(all_generated_sequences, dim=0)

    logging.info(
        f"Generated {generated_sequences_tensor.shape[0]} sequences "
        f"of length {sequence_length}."
    )
    return generated_sequences_tensor.numpy()


# --- Validation Function ---
def validate_synthetic_data(
    real_data_scaled, synthetic_data_scaled, feature_names,
    scaler, # Pass scaler to inverse transform for some plots
    num_features_to_plot=5, num_samples_for_tsne=1000,
    plot_save_dir="validation_plots"
):
    logging.info("Validating synthetic data...")

    output_dir = Path(plot_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if real_data_scaled.shape[1] != synthetic_data_scaled.shape[1]:
        raise ValueError(
            f"Real data features ({real_data_scaled.shape[1]}) != "
            f"Synthetic data features ({synthetic_data_scaled.shape[1]})"
        )

    num_features = real_data_scaled.shape[1]
    # Ensure num_features_to_plot is not more than available features
    num_features_to_plot = min(num_features_to_plot, num_features)
    plot_indices = np.linspace(
        0, num_features - 1, num_features_to_plot, dtype=int
    )

    # --- Basic Statistics (on scaled data) ---
    real_flat_scaled = real_data_scaled.reshape(-1, num_features)
    synth_flat_scaled = synthetic_data_scaled.reshape(-1, num_features)

    stats = {'real': {}, 'synthetic': {}}
    stats['real']['mean'] = np.mean(real_flat_scaled, axis=0)
    stats['real']['std'] = np.std(real_flat_scaled, axis=0)
    stats['synthetic']['mean'] = np.mean(synth_flat_scaled, axis=0)
    stats['synthetic']['std'] = np.std(synth_flat_scaled, axis=0)

    logging.info("--- Basic Statistics (Scaled Data [0, 1]) ---")
    header = (
        f"{'Feature':<15} | {'Real Mean':<10} | {'Synth Mean':<10} | "
        f"{'Real Std':<10} | {'Synth Std':<10}"
    )
    logging.info(header)
    logging.info("-" * len(header))
    for i in plot_indices:
        fname = feature_names[i][:15].ljust(15)
        logging.info(
            f"{fname} | {stats['real']['mean'][i]:<10.4f} | "
            f"{stats['synthetic']['mean'][i]:<10.4f} | "
            f"{stats['real']['std'][i]:<10.4f} | "
            f"{stats['synthetic']['std'][i]:<10.4f}"
        )

    # --- Distribution Plots (Histograms) ---
    logging.info(f"Plotting distributions for {num_features_to_plot} features...")
    plt.figure(figsize=(15, num_features_to_plot * 3))
    plt.suptitle("Feature Distribution Comparison (Scaled Data)", y=1.02)
    for i, idx in enumerate(plot_indices):
        plt.subplot(num_features_to_plot, 1, i + 1)
        plt.hist(
            real_flat_scaled[:, idx], bins=50, alpha=0.6,
            label='Real', density=True
        )
        plt.hist(
            synth_flat_scaled[:, idx], bins=50, alpha=0.6,
            label='Synthetic', density=True
        )
        plt.title(f"Distribution of {feature_names[idx]}")
        plt.legend()

        # Calculate Wasserstein distance
        try:
            dist = wasserstein_distance(
                real_flat_scaled[:, idx], synth_flat_scaled[:, idx]
            )
            logging.info(
                f"Wasserstein Distance for {feature_names[idx]} (scaled): {dist:.4f}"
            )
        except ValueError as e:
            logging.warning(
                f"Could not compute Wasserstein distance for "
                f"{feature_names[idx]}: {e}"
            )

    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
    hist_path = output_dir / "validation_histograms.png"
    plt.savefig(hist_path)
    logging.info(f"Validation histograms saved to {hist_path}")
    plt.close()

    # --- Autocorrelation (ACF) Plots ---
    logging.info(f"Plotting ACF for {num_features_to_plot} features...")
    # Inverse transform data for meaningful ACF plots
    # Use only a subset for performance if data is large
    sample_size_acf = min(len(real_data_scaled), 5000) # Limit samples for ACF calc
    real_data_orig = scaler.inverse_transform(real_data_scaled[:sample_size_acf])
    synth_data_orig = scaler.inverse_transform(synthetic_data_scaled[:sample_size_acf])


    plt.figure(figsize=(15, num_features_to_plot * 4))
    plt.suptitle("Autocorrelation Comparison (Original Scale)", y=1.01)
    max_lags = min(40, len(real_data_orig) // 2 - 1) # Sensible number of lags
    for i, idx in enumerate(plot_indices):
        ax1 = plt.subplot(num_features_to_plot, 2, 2 * i + 1)
        plot_acf(real_data_orig[:, idx], ax=ax1, title=f"Real ACF: {feature_names[idx]}", lags=max_lags)
        ax2 = plt.subplot(num_features_to_plot, 2, 2 * i + 2)
        plot_acf(synth_data_orig[:, idx], ax=ax2, title=f"Synthetic ACF: {feature_names[idx]}", lags=max_lags)
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
    acf_path = output_dir / "validation_acf.png"
    plt.savefig(acf_path)
    logging.info(f"Validation ACF plots saved to {acf_path}")
    plt.close()


    # --- t-SNE Visualization ---
    logging.info("Performing t-SNE analysis...")
    # Use a subset of data for t-SNE (it's computationally expensive)
    num_real_samples = real_flat_scaled.shape[0]
    num_synth_samples = synth_flat_scaled.shape[0]

    idx_real = np.random.permutation(num_real_samples)[:num_samples_for_tsne]
    idx_synth = np.random.permutation(num_synth_samples)[:num_samples_for_tsne]

    real_subset = real_flat_scaled[idx_real, :]
    synth_subset = synth_flat_scaled[idx_synth, :]

    # Combine data for joint t-SNE transformation
    combined_data = np.vstack((real_subset, synth_subset))
    labels = np.concatenate(
        (np.ones(len(real_subset)), np.zeros(len(synth_subset)))
    ) # 1 for Real, 0 for Synthetic

    try:
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(combined_data)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=labels, cmap='coolwarm', alpha=0.6
        )
        plt.title("t-SNE Visualization (Real vs. Synthetic)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(handles=scatter.legend_elements()[0], labels=['Synthetic', 'Real'])
        tsne_path = output_dir / "validation_tsne.png"
        plt.savefig(tsne_path)
        logging.info(f"t-SNE plot saved to {tsne_path}")
        plt.close()

    except Exception as e:
        logging.error(f"t-SNE calculation failed: {e}")


# --- Main Execution ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Data
    scaled_data, scaler, timestamps, feature_names = load_and_preprocess_data(
        args.data_dir, args.symbol, args.hdf_key, args.lob_depth
    )
    num_features = scaled_data.shape[1]
    # feature_names already loaded

    # 2. Create Dataset and Dataloader (using sequences)
    logging.info(f"Creating dataset with sequence length {args.sequence_length}")
    dataset = LOBDataset(scaled_data, args.sequence_length)
    # Adjust batch size if dataset is smaller
    effective_batch_size = min(args.batch_size, len(dataset))
    if effective_batch_size != args.batch_size:
        logging.warning(
            f"Batch size reduced to {effective_batch_size} "
            f"due to small dataset size ({len(dataset)} sequences)."
        )

    # Consider adding num_workers for faster loading if not on Windows or if using appropriate guards
    dataloader = DataLoader(
        dataset, batch_size=effective_batch_size, shuffle=True, num_workers=2 # Set num_workers
    )

    # 3. Initialize Model and Scheduler
    logging.info("Initializing Transformer Diffusion Model...")
    model = TransformerDiffusion(
        n_steps=args.diffusion_steps,
        num_features=num_features,
        sequence_length=args.sequence_length,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout,
    ).to(device)
    scheduler = DDPMScheduler(
        num_timesteps=args.diffusion_steps, device=device
    )

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model total trainable parameters: {total_params:,}")


    # 4. Train Model
    start_train_time = time.time()
    train_losses = train_diffusion_model(
        model, scheduler, dataloader, args.num_epochs, args.lr, device
    )
    logging.info(
        f"Training took {time.time() - start_train_time:.2f} seconds."
    )

    # Plot training loss
    loss_plot_path = Path("training_loss.png")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("Training Loss (Noise Prediction MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    logging.info(f"Training loss plot saved to {loss_plot_path}")

    # 5. Generate Synthetic Data
    start_gen_time = time.time()
    synthetic_sequences_orig = sample_diffusion_model(
        model, scheduler, args.num_samples, num_features,
        args.sequence_length, device, batch_size=args.sampling_batch_size
    )
    logging.info(
        f"Generation took {time.time() - start_gen_time:.2f} seconds."
    )

    # 6. Validate
    # Use the original scaled data (as sequences) and generated scaled sequences
    # Select a subset of real data for validation if it's too large
    max_real_seq_for_val = 10000 # Number of sequences
    val_real_indices = np.random.choice(
        len(scaled_data) - args.sequence_length + 1,
        min(max_real_seq_for_val, len(scaled_data) - args.sequence_length + 1),
        replace=False
    )
    # Extract sequences manually for validation subset
    real_sequences_scaled_for_val = np.array([
        scaled_data[i:i+args.sequence_length] for i in val_real_indices
    ])

    validate_synthetic_data(
        real_sequences_scaled_for_val, # Use sequences for validation if logic needs it
        synthetic_sequences_orig, # Already sequences
        feature_names,
        scaler, # Pass scaler
        num_features_to_plot=args.validation_num_features,
        num_samples_for_tsne=args.validation_num_tsne_samples,
        plot_save_dir=args.validation_plot_dir
    )

    # 8. Save Synthetic Data
    output_path = Path(args.output_hdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # synthetic_sequences_orig shape: (num_samples, seq_len, num_features)
    num_gen_samples, seq_len, num_features_gen = synthetic_sequences_orig.shape

    # Flatten for saving: (num_gen_samples * seq_len, num_features)
    synthetic_data_flat_orig = synthetic_sequences_orig.reshape(
        -1, num_features_gen
    )

    synthetic_df = pd.DataFrame(
        synthetic_data_flat_orig, columns=feature_names
    )

    # Add sequence_id column
    sequence_ids = np.repeat(np.arange(num_gen_samples), seq_len)
    synthetic_df.insert(0, 'sequence_id', sequence_ids)

    # Add a synthetic timestamp column
    if not timestamps.empty:
        # Increment from the last real timestamp, assuming 1-second intervals
        last_real_time = timestamps.iloc[-1]
        # Create range relative to number of *steps* generated
        synthetic_timestamps = pd.date_range(
            start=last_real_time + pd.Timedelta(seconds=1),
            periods=len(synthetic_df),
            freq='1S' # Or use detected frequency if available
        )
    else:
        # Start from current time if no real timestamps available
        start_time = pd.Timestamp.now(tz='UTC').floor('S')
        synthetic_timestamps = pd.date_range(
            start=start_time,
            periods=len(synthetic_df),
            freq='1S' # Default assumption
        )

    synthetic_df.insert(0, 'timestamp', synthetic_timestamps)

    logging.info(
        f"Saving {len(synthetic_df)} rows ({args.num_samples} sequences) "
        f"of synthetic data to {output_path} with key '{args.hdf_key}'"
    )
    try:
        # Use PyTables format for better performance with large datasets
        synthetic_df.to_hdf(
            output_path, key=args.hdf_key, mode='w', format='table'
        )
        logging.info("Synthetic data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save synthetic HDF5 file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Synthetic LOB Data using Transformer Diffusion Model"
    )
    # Data Args
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing real LOB HDF5 files"
    )
    parser.add_argument(
        "--output_hdf", type=str, required=True,
        help="Path to save the generated synthetic HDF5 file"
    )
    parser.add_argument(
        "--symbol", type=str, default="binance_BTCUSDT",
        help="Symbol part of the HDF5 filenames (e.g., 'binance_BTCUSDT')"
    )
    parser.add_argument(
        "--hdf_key", type=str, default="lob_data", help="HDF5 key for data"
    )
    parser.add_argument(
        "--lob_depth", type=int, default=10,
        help="Number of LOB levels (price/size pairs per side) used"
    )
    # Model Args
    parser.add_argument(
        "--sequence_length", type=int, default=64,
        help="Length of sequences for training and generation"
    )
    parser.add_argument(
        "--diffusion_steps", type=int, default=1000,
        help="Number of noise steps in DDPM"
    )
    parser.add_argument(
        "--transformer_d_model", type=int, default=128,
        help="Transformer model dimension"
    )
    parser.add_argument(
        "--transformer_nhead", type=int, default=8,
        help="Number of attention heads in Transformer"
    )
    parser.add_argument(
        "--transformer_num_layers", type=int, default=4,
        help="Number of Transformer encoder layers"
    )
    parser.add_argument(
        "--transformer_dim_feedforward", type=int, default=512,
        help="Dimension of Transformer feedforward layers"
    )
    parser.add_argument(
        "--transformer_dropout", type=float, default=0.1,
        help="Dropout rate in Transformer"
    )
    # Training Args
    parser.add_argument(
        "--num_epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training (number of sequences)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    # Sampling Args
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of synthetic sequences (samples) to generate"
    )
    parser.add_argument(
        "--sampling_batch_size", type=int, default=128,
        help="Batch size for generating samples (reduce if OOM)"
    )
    # Validation Args
    parser.add_argument(
        "--validation_num_features", type=int, default=5,
        help="Number of features to plot in validation"
    )
    parser.add_argument(
        "--validation_num_tsne_samples", type=int, default=1000,
        help="Number of samples (flattened steps) for t-SNE plot"
    )
    parser.add_argument(
        "--validation_plot_dir", type=str, default="validation_plots",
        help="Directory to save validation plots"
    )


    args = parser.parse_args()
    main(args)
