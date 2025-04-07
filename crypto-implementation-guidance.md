# Implementation Guide for Multi-Timeframe Crypto Trading Model

## Project Overview

We are building a cryptocurrency trading system that:
1. Uses OHLCV data from multiple timeframes (15m, 4h, 1d) aligned by timestamp
2. Progresses from a time series foundation model to a reinforcement learning agent
3. Includes synthetic data generation for robust pattern learning
4. Supports both long and short positions (trading "up and down")
5. Will eventually move to a live trading environment

## Required Project Structure

```
crypto_trading_model/
├── data_processing/
│   ├── __init__.py
│   ├── data_loader.py          # For loading historical OHLCV data
│   ├── feature_engineering.py  # For creating technical indicators across timeframes
│   └── data_alignment.py       # For aligning multiple timeframes by timestamp
├── synthetic_data/
│   ├── __init__.py
│   ├── pattern_generator.py    # Core synthetic pattern generation
│   ├── indicator_engineering.py # Generate realistic indicators for synthetic data
│   ├── multi_timeframe_sync.py # Ensure consistent patterns across timeframes
│   └── dataset_builder.py      # Create comprehensive training datasets
├── models/
│   ├── __init__.py
│   ├── time_series/
│   │   ├── __init__.py
│   │   ├── model.py            # Multi-timeframe time series model architecture
│   │   └── trainer.py          # Training loop and evaluation for time series model
│   └── reinforcement/
│       ├── __init__.py
│       ├── trading_env.py      # OpenAI Gym environment for crypto trading
│       ├── dqn_agent.py        # Initial DQN implementation
│       ├── ppo_agent.py        # Advanced PPO implementation
│       └── reward_functions.py # Different reward strategies for RL
├── evaluation/
│   ├── __init__.py
│   ├── backtest.py             # Backtesting framework
│   └── metrics.py              # Trading performance metrics
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # For plotting results
│   └── config.py               # Configuration parameters
├── requirements.txt            # All dependencies
└── main.py                     # Entry point for running training/evaluation
```

## Key Components Implementation

### 1. Data Processing

#### Data Loader (`data_processing/data_loader.py`)
```python
def load_historical_data(symbols, timeframes, start_date, end_date):
    """
    Load historical OHLCV data for given symbols and timeframes
    
    Parameters:
    - symbols: List of cryptocurrency symbols to load
    - timeframes: List of timeframes to load (e.g., ["15m", "4h", "1d"])
    - start_date: Start date for historical data
    - end_date: End date for historical data
    
    Returns:
    - Dictionary of DataFrames with OHLCV data, keyed by symbol and timeframe
    """
    pass
```

#### Data Alignment (`data_processing/data_alignment.py`)
```python
def align_timeframes(data_dict):
    """
    Align data from multiple timeframes by timestamp
    
    Parameters:
    - data_dict: Dictionary of DataFrames with different timeframes
    
    Returns:
    - Dictionary of aligned DataFrames with same timestamp index
    """
    pass
```

#### Feature Engineering (`data_processing/feature_engineering.py`)
```python
def calculate_technical_indicators(df):
    """
    Calculate technical indicators for a DataFrame
    
    Parameters:
    - df: DataFrame with OHLCV data
    
    Returns:
    - DataFrame with added technical indicators
    """
    # Implement indicators like:
    # - Moving averages (SMA, EMA)
    # - RSI, MACD, Bollinger Bands
    # - Volume indicators
    # - Support/resistance levels
    pass

def create_multi_timeframe_features(aligned_data):
    """
    Create features that capture relationships between timeframes
    
    Parameters:
    - aligned_data: Dictionary of aligned DataFrames for different timeframes
    
    Returns:
    - DataFrame with features from all timeframes
    """
    pass
```

### 2. Synthetic Data Generation

#### Pattern Generator (`synthetic_data/pattern_generator.py`)
```python
def generate_trend_pattern(length, trend_type, noise_level=0.05, volatility_profile="medium"):
    """
    Generate synthetic price data with trend pattern
    
    Parameters:
    - length: Number of candles to generate
    - trend_type: Type of trend ('uptrend', 'downtrend', 'sideways')
    - noise_level: Amount of noise to add
    - volatility_profile: Volatility characteristics ("low", "medium", "high")
    
    Returns:
    - DataFrame with OHLCV data containing the trend pattern
    """
    pass

def generate_reversal_pattern(length, pattern_type, noise_level=0.05, volume_profile="increasing"):
    """
    Generate synthetic price data with reversal pattern
    
    Parameters:
    - length: Number of candles to generate
    - pattern_type: Type of reversal pattern ('double_top', 'head_shoulders', etc.)
    - noise_level: Amount of noise to add
    - volume_profile: Volume behavior during pattern formation
    
    Returns:
    - DataFrame with OHLCV data containing the reversal pattern
    """
    pass

def generate_support_resistance_reaction(length, reaction_type, strength="strong", noise_level=0.05):
    """
    Generate synthetic price data reacting to support/resistance
    
    Parameters:
    - length: Number of candles to generate
    - reaction_type: Type of reaction ('bounce', 'breakout')
    - strength: Strength of support/resistance level
    - noise_level: Amount of noise to add
    
    Returns:
    - DataFrame with OHLCV data showing support/resistance reaction
    """
    pass

def add_realistic_noise(price_data, noise_profile="market_like"):
    """
    Add realistic market-like noise to synthetic price data
    
    Parameters:
    - price_data: Clean synthetic price data
    - noise_profile: Type of noise to add ("market_like", "choppy", "trending")
    
    Returns:
    - DataFrame with realistic market noise added
    """
    # Add non-uniform noise that mimics real market behavior
    # Apply different noise profiles for different market conditions
    # Ensure OHLC relationships remain valid
    pass

def create_realistic_volume_profile(price_data, pattern_type):
    """
    Create realistic volume profile based on price pattern
    
    Parameters:
    - price_data: Synthetic price data
    - pattern_type: Type of pattern in the data
    
    Returns:
    - Series with volume data that matches price behavior
    """
    # Generate volume that corresponds to price movements
    # Create appropriate volume spikes at key levels
    # Ensure volume precedes price in breakouts
    # Add declining volume in consolidations
    pass
```

#### Indicator Engineering (`synthetic_data/indicator_engineering.py`)
```python
def calculate_realistic_indicators(synthetic_price_data):
    """
    Calculate technical indicators for synthetic data that behave realistically
    
    Parameters:
    - synthetic_price_data: DataFrame with synthetic OHLCV data
    
    Returns:
    - DataFrame with added technical indicators that behave as they would in real markets
    """
    # Calculate indicators ensuring they reflect the underlying pattern
    # Ensure indicator relationships make sense (e.g., MACD crosses during trend changes)
    # Add appropriate noise to indicators to mimic real market behavior
    pass

def generate_pattern_with_indicators(pattern_type, params, include_indicators=True):
    """
    Generate synthetic pattern with corresponding indicator behavior
    
    Parameters:
    - pattern_type: Type of pattern to generate
    - params: Parameters for pattern generation
    - include_indicators: Whether to include technical indicators
    
    Returns:
    - DataFrame with price data and indicators
    """
    # Generate base price pattern
    # Calculate realistic indicators that match the pattern
    # Ensure indicator confluences that would realistically occur with the pattern
    pass

def create_feature_confluences(base_pattern, confluence_type):
    """
    Create specific technical indicator confluences in synthetic data
    
    Parameters:
    - base_pattern: Base price pattern data
    - confluence_type: Type of confluence to create (e.g., 'bullish_divergence', 'support_test')
    
    Returns:
    - Modified pattern with specific indicator confluences
    """
    # Modify the base pattern to include specific indicator confluences
    # Ensure indicators react appropriately to price movements
    # Create realistic relationships between multiple indicators
    pass

def simulate_indicator_lag(price_data, indicators):
    """
    Simulate realistic lag in indicators relative to price
    
    Parameters:
    - price_data: Synthetic price data
    - indicators: Dictionary of calculated indicators
    
    Returns:
    - Modified indicators with realistic lag characteristics
    """
    # Add appropriate lag to lagging indicators
    # Ensure leading indicators actually lead with realistic characteristics
    # Create proper relationship between price and indicator movements
    pass

def add_indicator_noise(indicators, noise_level="realistic"):
    """
    Add realistic noise to technical indicators
    
    Parameters:
    - indicators: Dictionary of calculated indicators
    - noise_level: Amount and type of noise to add
    
    Returns:
    - Modified indicators with realistic noise characteristics
    """
    # Add non-uniform noise to indicators
    # Ensure noise doesn't invalidate the signal
    # Create realistic false signals at appropriate frequency
    pass
```

#### Multi-Timeframe Synthetic Data (`synthetic_data/multi_timeframe_sync.py`)
```python
def generate_multi_timeframe_data(pattern_params, timeframes=["15m", "4h", "1d"]):
    """
    Generate consistent synthetic data across multiple timeframes
    
    Parameters:
    - pattern_params: Parameters for pattern generation
    - timeframes: List of timeframes to generate
    
    Returns:
    - Dictionary of DataFrames for each timeframe with consistent patterns
    """
    # Generate highest resolution data first
    # Aggregate to create higher timeframe data
    # Ensure patterns are visible across timeframes consistently
    pass

def ensure_pattern_visibility(multi_tf_data, pattern_type, primary_timeframe):
    """
    Ensure pattern is appropriately visible across timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - pattern_type: Type of pattern generated
    - primary_timeframe: Timeframe where pattern should be most obvious
    
    Returns:
    - Modified multi-timeframe data with properly visible patterns
    """
    # Adjust data to ensure pattern is visible on primary timeframe
    # Make sure pattern has appropriate expression in other timeframes
    # Maintain realistic relationships between timeframes
    pass

def create_timeframe_confluences(multi_tf_data, confluence_type):
    """
    Create confluences between different timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - confluence_type: Type of confluence to create
    
    Returns:
    - Modified multi-timeframe data with specific confluences
    """
    # Create specific confluences between timeframes
    # Ensure indicators align across timeframes at key points
    # Create realistic divergences and confirmations between timeframes
    pass
```

#### Dataset Builder (`synthetic_data/dataset_builder.py`)
```python
def build_synthetic_dataset(num_samples, pattern_distribution, with_indicators=True):
    """
    Build a complete synthetic dataset with diverse patterns
    
    Parameters:
    - num_samples: Number of samples to generate
    - pattern_distribution: Dictionary defining distribution of patterns
    - with_indicators: Whether to include technical indicators
    
    Returns:
    - DataFrames for training with labels for expected actions
    """
    # Generate diverse patterns according to distribution
    # Include various market conditions
    # Add labels for supervised learning
    pass

def create_adversarial_examples(base_dataset, num_adversarial):
    """
    Create adversarial examples that look like common patterns but behave differently
    
    Parameters:
    - base_dataset: Original synthetic dataset
    - num_adversarial: Number of adversarial examples to create
    
    Returns:
    - DataFrame with adversarial examples
    """
    # Create examples that look like common patterns but resolve differently
    # Include false breakouts, failed patterns, etc.
    # Label appropriately for model robustness
    pass

def blend_synthetic_with_real(synthetic_data, real_data, blend_ratio=0.5):
    """
    Blend synthetic data with real market data
    
    Parameters:
    - synthetic_data: Synthetic dataset
    - real_data: Real market data
    - blend_ratio: Ratio of synthetic to real data
    
    Returns:
    - Blended dataset that maintains realistic characteristics
    """
    # Combine synthetic and real data
    # Ensure seamless transitions
    # Preserve pattern characteristics
    pass
```

### 3. Time Series Model

#### Model Architecture (`models/time_series/model.py`)
```python
class MultiTimeframeModel(nn.Module):
    """
    Deep learning model that processes multiple timeframes
    
    Features:
    - Separate input paths for each timeframe
    - LSTM/GRU layers for sequence processing
    - Attention mechanisms between timeframes
    - Dense output layers for predictions
    """
    
    def __init__(self, input_dims, hidden_dims, num_layers):
        super().__init__()
        # Initialize layers for each timeframe
        # Create fusion mechanism
        # Define output layers
        
    def forward(self, x_15m, x_4h, x_1d):
        # Process each timeframe
        # Fuse information
        # Generate predictions
        pass
```

#### Model Trainer (`models/time_series/trainer.py`)
```python
def train_time_series_model(model, train_data, val_data, epochs, batch_size, learning_rate):
    """
    Train the time series model
    
    Parameters:
    - model: Model instance to train
    - train_data: Training data (features and targets)
    - val_data: Validation data
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    
    Returns:
    - Trained model and training history
    """
    # Setup optimizer and loss function
    # Training loop with validation
    # Early stopping
    # Learning rate scheduling
    pass
```

### 4. Reinforcement Learning

#### Trading Environment (`models/reinforcement/trading_env.py`)
```python
class CryptoTradingEnv(gym.Env):
    """
    Trading environment for cryptocurrency
    
    Features:
    - Support for multiple timeframes
    - Long and short positions
    - Transaction costs
    - Realistic market constraints
    """
    
    def __init__(self, data, window_size=50, transaction_cost=0.001):
        super().__init__()
        # Initialize with market data
        # Define action and observation spaces
        # Setup initial state
        
    def reset(self):
        # Reset to initial state
        # Return observation
        pass
        
    def step(self, action):
        # Process action (buy/sell/hold)
        # Update position
        # Calculate reward
        # Update state
        # Return observation, reward, done, info
        pass
```

#### DQN Agent (`models/reinforcement/dqn_agent.py`)
```python
class DQNAgent:
    """
    Deep Q-Network agent for trading
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=0.001):
        # Initialize Q-networks
        # Setup replay buffer
        # Define optimizer
        
    def select_action(self, state, epsilon):
        # Epsilon-greedy action selection
        pass
        
    def train(self, batch):
        # Sample batch from replay buffer
        # Calculate target Q-values
        # Update network
        pass
```

#### PPO Agent (`models/reinforcement/ppo_agent.py`)
```python
class PPOAgent:
    """
    Proximal Policy Optimization agent for trading
    
    Features:
    - Actor-critic architecture
    - Clipped objective function
    - Value function estimation
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=0.0003):
        # Initialize actor and critic networks
        # Setup optimizers
        
    def select_action(self, state):
        # Sample action from policy
        pass
        
    def train(self, states, actions, rewards, next_states, dones):
        # Compute advantages
        # Update policy and value function
        pass
```

#### Reward Functions (`models/reinforcement/reward_functions.py`)
```python
def pnl_reward(action, position, next_price, current_price, transaction_cost):
    """
    Calculate reward based on profit/loss
    
    Parameters:
    - action: Action taken by agent
    - position: Current position
    - next_price: Next price
    - current_price: Current price
    - transaction_cost: Transaction cost percentage
    
    Returns:
    - Reward value
    """
    pass

def risk_adjusted_reward(returns, risk_free_rate=0.0):
    """
    Calculate risk-adjusted reward (Sharpe-like)
    
    Parameters:
    - returns: Array of returns
    - risk_free_rate: Risk-free rate
    
    Returns:
    - Risk-adjusted reward
    """
    pass
```

### 5. Evaluation

#### Backtesting (`evaluation/backtest.py`)
```python
def backtest_strategy(model, test_data, initial_capital=10000.0, transaction_cost=0.001):
    """
    Backtest a trading strategy
    
    Parameters:
    - model: Trained model (time series or RL)
    - test_data: Test data for backtesting
    - initial_capital: Initial capital
    - transaction_cost: Transaction cost percentage
    
    Returns:
    - DataFrame with trades and performance metrics
    """
    pass
```

#### Performance Metrics (`evaluation/metrics.py`)
```python
def calculate_performance_metrics(returns, trades):
    """
    Calculate trading performance metrics
    
    Parameters:
    - returns: Array of returns
    - trades: DataFrame with trade information
    
    Returns:
    - Dictionary of performance metrics
    """
    # Calculate metrics like:
    # - Total return, annualized return
    # - Sharpe ratio, Sortino ratio
    # - Maximum drawdown
    # - Win rate, profit factor
    pass
```

## Implementation Steps

1. **Initial Setup**:
   - Create the project structure
   - Install dependencies
   - Setup configuration

2. **Data Processing Pipeline**:
   - Implement data loading for historical data
   - Create feature engineering functions
   - Build multi-timeframe alignment

3. **Synthetic Data Generation**:
   - Implement basic pattern generators
   - Create realistic indicator behavior
   - Build multi-timeframe synchronization
   - Develop comprehensive dataset generator with pattern diversity

4. **Time Series Model**:
   - Implement model architecture
   - Create training pipeline
   - Train and evaluate on synthetic then historical data

5. **RL Environment**:
   - Build trading environment
   - Implement reward functions
   - Create evaluation methods

6. **DQN Implementation**:
   - Build DQN agent
   - Train on simpler scenarios
   - Evaluate performance

7. **PPO Implementation**:
   - Build PPO agent
   - Transfer learning from time series model
   - Train and evaluate

8. **Performance Evaluation**:
   - Implement backtesting framework
   - Create visualization tools
   - Compare models and strategies

## Guidelines for Claude

1. Follow the exact project structure outlined above.
2. Implement each component step by step.
3. Ensure all classes and functions have proper docstrings.
4. Use type hints for better code readability.
5. Implement proper error handling.
6. Ensure consistency between the multi-timeframe data processing.
7. Use configuration files for hyperparameters.
8. Include logging throughout the system.
9. Keep the implementation modular and extensible.
10. Include proper validation at each step.
11. Ensure the reinforcement learning environment follows OpenAI Gym standards.
12. Make sure the synthetic data generator creates realistic patterns with appropriate indicator behavior.
13. Remove any unused or mistakenly created files to keep the environment clean.

## Environment and Dependencies

- Python 3.8+
- PyTorch for deep learning components
- Pandas and NumPy for data processing
- Gym for reinforcement learning environment
- TA-Lib or custom implementations for technical indicators
- Matplotlib and Plotly for visualization

## Final Checklist

Before considering the implementation complete, verify:

- [ ] All components are implemented according to specifications
- [ ] Code is well-documented
- [ ] Error handling is in place
- [ ] Multi-timeframe alignment is working correctly
- [ ] Synthetic data generation creates realistic patterns with proper indicator behavior
- [ ] Synthetic data is virtually indistinguishable from real data except for controlled patterns
- [ ] Time series model can process multiple timeframes
- [ ] RL environment supports both long and short positions
- [ ] DQN and PPO implementations follow best practices
- [ ] Performance metrics are comprehensive
- [ ] Visualization tools are informative
- [ ] No unused or mistakenly created files are present
