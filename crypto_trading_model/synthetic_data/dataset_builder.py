import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_synthetic_dataset(num_samples: int, 
                           pattern_distribution: Dict[str, float],
                           with_indicators: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Build a complete synthetic dataset with diverse patterns
    
    Parameters:
    - num_samples: Number of samples to generate
    - pattern_distribution: Dictionary defining distribution of patterns
    - with_indicators: Whether to include technical indicators
    
    Returns:
    - DataFrames for training with labels for expected actions
    """
    from .pattern_generator import (generate_trend_pattern, 
                                   generate_reversal_pattern,
                                   generate_support_resistance_reaction,
                                   add_realistic_noise)
    from .indicator_engineering import calculate_realistic_indicators
    
    # Normalize the pattern distribution
    total_weight = sum(pattern_distribution.values())
    normalized_dist = {k: v / total_weight for k, v in pattern_distribution.items()}
    
    # Calculate the number of samples for each pattern
    pattern_counts = {}
    remaining = num_samples
    
    for pattern, weight in normalized_dist.items():
        if pattern == list(normalized_dist.keys())[-1]:
            # Last pattern gets any remaining samples
            pattern_counts[pattern] = remaining
        else:
            count = int(num_samples * weight)
            pattern_counts[pattern] = count
            remaining -= count
    
    logger.info(f"Building dataset with {num_samples} samples: {pattern_counts}")
    
    # Generate samples for each pattern type
    all_samples = []
    labels = []
    
    for pattern, count in pattern_counts.items():
        logger.info(f"Generating {count} samples for pattern '{pattern}'")
        
        for i in range(count):
            # Generate sample based on pattern type
            if 'trend' in pattern.lower():
                # Handle different trend types
                if 'up' in pattern.lower():
                    trend_type = 'uptrend'
                    expected_action = 1  # Buy/long
                elif 'down' in pattern.lower():
                    trend_type = 'downtrend'
                    expected_action = -1  # Sell/short
                else:
                    trend_type = 'sideways'
                    expected_action = 0  # Hold/neutral
                
                # Random length between 80 and 120 candles
                length = random.randint(80, 120)
                
                # Random volatility profile
                volatility = random.choice(['low', 'medium', 'high'])
                
                # Generate the sample
                sample = generate_trend_pattern(
                    length=length,
                    trend_type=trend_type,
                    noise_level=random.uniform(0.03, 0.07),
                    volatility_profile=volatility
                )
                
            elif 'reversal' in pattern.lower():
                # Handle different reversal types
                if 'top' in pattern.lower() or 'head' in pattern.lower():
                    reversal_type = random.choice(['double_top', 'head_shoulders'])
                    expected_action = -1  # Sell/short
                else:
                    reversal_type = 'v_reversal'
                    expected_action = 1  # Buy/long
                
                # Random length between 80 and 120 candles
                length = random.randint(80, 120)
                
                # Generate the sample
                sample = generate_reversal_pattern(
                    length=length,
                    pattern_type=reversal_type,
                    noise_level=random.uniform(0.03, 0.07),
                    volume_profile="increasing"
                )
                
            elif 'support' in pattern.lower() or 'resistance' in pattern.lower():
                # Handle support/resistance patterns
                if 'bounce' in pattern.lower():
                    reaction_type = 'bounce'
                    expected_action = 1  # Buy/long
                else:
                    reaction_type = 'breakout'
                    expected_action = random.choice([1, -1])  # Can be either bullish or bearish breakout
                
                # Random length between 80 and 120 candles
                length = random.randint(80, 120)
                
                # Generate the sample
                sample = generate_support_resistance_reaction(
                    length=length,
                    reaction_type=reaction_type,
                    strength=random.choice(['weak', 'medium', 'strong']),
                    noise_level=random.uniform(0.03, 0.07)
                )
                
            else:
                # Default to random uptrend
                logger.warning(f"Unknown pattern type '{pattern}', defaulting to uptrend")
                sample = generate_trend_pattern(
                    length=random.randint(80, 120),
                    trend_type='uptrend',
                    noise_level=random.uniform(0.03, 0.07)
                )
                expected_action = 1  # Buy/long
            
            # Add realistic noise to make the pattern more subtle
            sample = add_realistic_noise(sample, noise_profile=random.choice([
                "market_like", "choppy", "trending"
            ]))
            
            # Add technical indicators if requested
            if with_indicators:
                sample = calculate_realistic_indicators(sample)
            
            # Add to the collection
            all_samples.append(sample)
            labels.append(expected_action)
    
    # Create a dictionary with the dataset and labels
    result = {
        'data': all_samples,
        'labels': pd.Series(labels)
    }
    
    return result

def create_adversarial_examples(base_dataset: Dict[str, Union[List[pd.DataFrame], pd.Series]], 
                               num_adversarial: int) -> Dict[str, Union[List[pd.DataFrame], pd.Series]]:
    """
    Create adversarial examples that look like common patterns but behave differently
    
    Parameters:
    - base_dataset: Original synthetic dataset
    - num_adversarial: Number of adversarial examples to create
    
    Returns:
    - DataFrame with adversarial examples
    """
    from .pattern_generator import add_realistic_noise
    
    # Extract data and labels
    data = base_dataset['data']
    labels = base_dataset['labels']
    
    # Create copies for adversarial examples
    adversarial_data = []
    adversarial_labels = []
    
    # Select random samples to create adversarial versions
    sample_indices = np.random.choice(range(len(data)), size=num_adversarial, replace=False)
    
    for idx in sample_indices:
        original_sample = data[idx].copy()
        original_label = labels.iloc[idx]
        
        # Choose an adversarial modification type
        mod_type = random.choice([
            'false_breakout',
            'failed_pattern',
            'late_reversal'
        ])
        
        # Apply the modification
        if mod_type == 'false_breakout':
            # Create a false breakout that reverses
            # Identify the end of the pattern (last 20% of candles)
            length = len(original_sample)
            breakout_start = int(length * 0.8)
            
            # Create initial breakout in the direction expected
            for i in range(breakout_start, min(breakout_start + 5, length)):
                # Create strong breakout candle
                if original_label > 0:  # Expected bullish
                    original_sample.iloc[i]['close'] *= 1.02  # 2% up
                    original_sample.iloc[i]['high'] = max(original_sample.iloc[i]['high'], 
                                                         original_sample.iloc[i]['close'] * 1.01)
                else:  # Expected bearish
                    original_sample.iloc[i]['close'] *= 0.98  # 2% down
                    original_sample.iloc[i]['low'] = min(original_sample.iloc[i]['low'], 
                                                        original_sample.iloc[i]['close'] * 0.99)
            
            # Then reverse the direction for the remaining candles
            for i in range(breakout_start + 5, length):
                # Create reversal candles
                if original_label > 0:  # Expected bullish but now bearish
                    original_sample.iloc[i]['close'] *= (0.99 - 0.003 * (i - breakout_start - 5))
                    original_sample.iloc[i]['low'] = min(original_sample.iloc[i]['low'], 
                                                        original_sample.iloc[i]['close'] * 0.99)
                else:  # Expected bearish but now bullish
                    original_sample.iloc[i]['close'] *= (1.01 + 0.003 * (i - breakout_start - 5))
                    original_sample.iloc[i]['high'] = max(original_sample.iloc[i]['high'], 
                                                         original_sample.iloc[i]['close'] * 1.01)
            
            # Reverse the label
            adversarial_label = -original_label
            
        elif mod_type == 'failed_pattern':
            # Make a pattern fail to complete as expected
            length = len(original_sample)
            failure_point = int(length * 0.7)
            
            # For the last 30% of candles, move sideways instead of completing the pattern
            base_price = original_sample.iloc[failure_point]['close']
            
            for i in range(failure_point, length):
                # Replace with sideways price action
                original_sample.iloc[i]['close'] = base_price * (1 + np.random.normal(0, 0.005))
                original_sample.iloc[i]['high'] = original_sample.iloc[i]['close'] * (1 + np.random.uniform(0.001, 0.003))
                original_sample.iloc[i]['low'] = original_sample.iloc[i]['close'] * (1 - np.random.uniform(0.001, 0.003))
                original_sample.iloc[i]['open'] = base_price * (1 + np.random.normal(0, 0.005))
            
            # Change to neutral
            adversarial_label = 0
            
        elif mod_type == 'late_reversal':
            # Pattern seems to complete but then reverses late
            length = len(original_sample)
            reversal_point = int(length * 0.9)
            
            # Let the pattern complete almost entirely
            # Then add a strong reversal at the end
            for i in range(reversal_point, length):
                # Add a sharp reversal
                if original_label > 0:  # Expected bullish, add bearish reversal
                    change_factor = 0.99 - 0.01 * (i - reversal_point)
                    original_sample.iloc[i]['close'] = original_sample.iloc[i-1]['close'] * change_factor
                    original_sample.iloc[i]['low'] = original_sample.iloc[i]['close'] * 0.99
                    original_sample.iloc[i]['high'] = original_sample.iloc[i-1]['close'] * 1.003
                    original_sample.iloc[i]['open'] = original_sample.iloc[i-1]['close'] * 1.001
                else:  # Expected bearish, add bullish reversal
                    change_factor = 1.01 + 0.01 * (i - reversal_point)
                    original_sample.iloc[i]['close'] = original_sample.iloc[i-1]['close'] * change_factor
                    original_sample.iloc[i]['high'] = original_sample.iloc[i]['close'] * 1.01
                    original_sample.iloc[i]['low'] = original_sample.iloc[i-1]['close'] * 0.997
                    original_sample.iloc[i]['open'] = original_sample.iloc[i-1]['close'] * 0.999
            
            # Reverse the label
            adversarial_label = -original_label
        
        # Add realistic noise to make the adversarial pattern look natural
        adversarial_sample = add_realistic_noise(original_sample, noise_profile="market_like")
        
        # If the sample contains indicators, update them to match the new price pattern
        if 'rsi_14' in adversarial_sample.columns:
            from .indicator_engineering import calculate_realistic_indicators
            adversarial_sample = calculate_realistic_indicators(adversarial_sample)
        
        # Add to the collection
        adversarial_data.append(adversarial_sample)
        adversarial_labels.append(adversarial_label)
    
    # Create a dictionary with the adversarial examples
    result = {
        'data': adversarial_data,
        'labels': pd.Series(adversarial_labels)
    }
    
    return result

def blend_synthetic_with_real(synthetic_data: Dict[str, Union[List[pd.DataFrame], pd.Series]], 
                             real_data: Dict[str, Union[List[pd.DataFrame], pd.Series]], 
                             blend_ratio: float = 0.5) -> Dict[str, Union[List[pd.DataFrame], pd.Series]]:
    """
    Blend synthetic data with real market data
    
    Parameters:
    - synthetic_data: Synthetic dataset
    - real_data: Real market data
    - blend_ratio: Ratio of synthetic to real data
    
    Returns:
    - Blended dataset that maintains realistic characteristics
    """
    # Calculate number of samples from each source
    total_samples = len(synthetic_data['data']) + len(real_data['data'])
    synthetic_samples = int(total_samples * blend_ratio)
    real_samples = total_samples - synthetic_samples
    
    # Ensure we don't try to take more samples than available
    synthetic_samples = min(synthetic_samples, len(synthetic_data['data']))
    real_samples = min(real_samples, len(real_data['data']))
    
    # Select random samples from each source
    synthetic_indices = np.random.choice(
        range(len(synthetic_data['data'])), 
        size=synthetic_samples, 
        replace=False
    )
    
    real_indices = np.random.choice(
        range(len(real_data['data'])), 
        size=real_samples, 
        replace=False
    )
    
    # Combine the selected samples
    blended_data = []
    blended_labels = []
    
    for idx in synthetic_indices:
        blended_data.append(synthetic_data['data'][idx])
        blended_labels.append(synthetic_data['labels'].iloc[idx])
    
    for idx in real_indices:
        blended_data.append(real_data['data'][idx])
        blended_labels.append(real_data['labels'].iloc[idx])
    
    # Shuffle the combined data
    combined = list(zip(blended_data, blended_labels))
    random.shuffle(combined)
    blended_data, blended_labels = zip(*combined)
    
    # Create the blended dataset
    result = {
        'data': list(blended_data),
        'labels': pd.Series(blended_labels)
    }
    
    return result

def save_dataset(dataset: Dict[str, Union[List[pd.DataFrame], pd.Series]], 
                output_dir: str,
                dataset_name: str = "synthetic_dataset") -> None:
    """
    Save a dataset to disk
    
    Parameters:
    - dataset: Dataset dictionary with 'data' and 'labels'
    - output_dir: Directory to save the dataset
    - dataset_name: Base name for the dataset files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each data sample as a separate CSV file
    data_dir = os.path.join(output_dir, f"{dataset_name}_data")
    os.makedirs(data_dir, exist_ok=True)
    
    for i, df in enumerate(dataset['data']):
        file_path = os.path.join(data_dir, f"sample_{i:04d}.csv")
        df.to_csv(file_path)
    
    # Save labels as a single CSV
    labels_path = os.path.join(output_dir, f"{dataset_name}_labels.csv")
    dataset['labels'].to_csv(labels_path, header=['label'], index_label='sample_id')
    
    logger.info(f"Dataset saved to {output_dir}")

def load_dataset(dataset_dir: str, 
                dataset_name: str = "synthetic_dataset") -> Dict[str, Union[List[pd.DataFrame], pd.Series]]:
    """
    Load a dataset from disk
    
    Parameters:
    - dataset_dir: Directory containing the dataset
    - dataset_name: Base name of the dataset files
    
    Returns:
    - Loaded dataset dictionary with 'data' and 'labels'
    """
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} not found")
        return {'data': [], 'labels': pd.Series([])}
    
    # Load labels
    labels_path = os.path.join(dataset_dir, f"{dataset_name}_labels.csv")
    if not os.path.exists(labels_path):
        logger.error(f"Labels file {labels_path} not found")
        return {'data': [], 'labels': pd.Series([])}
    
    labels = pd.read_csv(labels_path, index_col='sample_id')['label']
    
    # Load data samples
    data_dir = os.path.join(dataset_dir, f"{dataset_name}_data")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found")
        return {'data': [], 'labels': labels}
    
    # Get all CSV files in the data directory
    sample_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    # Load each data sample
    data = []
    for file_name in sample_files:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data.append(df)
    
    logger.info(f"Loaded dataset with {len(data)} samples from {dataset_dir}")
    
    return {'data': data, 'labels': labels}

def create_train_val_test_split(dataset: Dict[str, Union[List[pd.DataFrame], pd.Series]], 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               shuffle: bool = True) -> Dict[str, Dict[str, Union[List[pd.DataFrame], pd.Series]]]:
    """
    Split a dataset into training, validation, and test sets
    
    Parameters:
    - dataset: Dataset dictionary with 'data' and 'labels'
    - train_ratio: Ratio of data for training set
    - val_ratio: Ratio of data for validation set
    - test_ratio: Ratio of data for test set
    - shuffle: Whether to shuffle the data before splitting
    
    Returns:
    - Dictionary with 'train', 'val', and 'test' splits
    """
    # Check if ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        logger.warning(f"Split ratios ({train_ratio}, {val_ratio}, {test_ratio}) don't sum to 1, normalizing")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Create combined list of data and labels
    combined = list(zip(dataset['data'], dataset['labels']))
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(combined)
    
    # Calculate split indices
    n_samples = len(combined)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split the data
    train_data, train_labels = zip(*combined[:train_end]) if train_end > 0 else ([], [])
    val_data, val_labels = zip(*combined[train_end:val_end]) if val_end > train_end else ([], [])
    test_data, test_labels = zip(*combined[val_end:]) if val_end < n_samples else ([], [])
    
    # Create result dictionary
    result = {
        'train': {
            'data': list(train_data),
            'labels': pd.Series(train_labels)
        },
        'val': {
            'data': list(val_data),
            'labels': pd.Series(val_labels)
        },
        'test': {
            'data': list(test_data),
            'labels': pd.Series(test_labels)
        }
    }
    
    logger.info(f"Created dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    
    return result 