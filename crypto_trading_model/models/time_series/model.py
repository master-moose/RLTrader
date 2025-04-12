import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (seq_len, batch_size, hidden_dim)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attn_output))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = SelfAttention(hidden_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask)
        x = self.norm(x + self.ff(x))
        return x

class MultiTimeframeModel(nn.Module):
    """
    Deep learning model that processes multiple timeframes
    
    Features:
    - Separate input paths for each timeframe
    - LSTM/GRU layers for sequence processing
    - Attention mechanisms between timeframes
    - Dense output layers for predictions
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dims: int = 64,
        num_layers: int = 1,
        dropout: float = 0.7,
        bidirectional: bool = True,
        attention: bool = False,
        num_classes: int = 3,
        use_batch_norm: bool = True,
        num_heads: int = 8,
        use_transformer: bool = False
    ):
        """
        Initialize the multi-timeframe model
        
        Parameters:
        - input_dims: Dictionary mapping timeframe names to input feature dimensions
        - hidden_dims: Dimension of hidden layers
        - num_layers: Number of LSTM/GRU layers
        - dropout: Dropout probability
        - bidirectional: Whether to use bidirectional LSTM/GRU
        - attention: Whether to use attention mechanism
        - num_classes: Number of output classes (default: 3 for buy, sell, hold)
        - use_batch_norm: Whether to use batch normalization
        - num_heads: Number of attention heads for transformer architecture
        - use_transformer: Whether to use transformer architecture
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        self.use_transformer = use_transformer
        
        # Input processing for each timeframe
        self.input_layers = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(dim, hidden_dims),
                nn.BatchNorm1d(hidden_dims) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for tf, dim in input_dims.items()
        })
        
        if use_transformer:
            # Transformer-based architecture
            self.pos_encoding = PositionalEncoding(hidden_dims)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(hidden_dims, num_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            # LSTM-based architecture
            self.lstm_layers = nn.ModuleDict({
                tf: nn.LSTM(
                    input_size=hidden_dims,
                    hidden_size=hidden_dims,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0
                )
                for tf in input_dims.keys()
            })
            
            if attention:
                self.attention_layers = nn.ModuleDict({
                    tf: SelfAttention(hidden_dims * (2 if bidirectional else 1), num_heads, dropout)
                    for tf in input_dims.keys()
                })
        
        # Output layers
        final_hidden_dim = hidden_dims * (2 if bidirectional else 1) * len(input_dims)
        self.output_layers = nn.Sequential(
            nn.Linear(final_hidden_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, num_classes)
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model
        
        Parameters:
        - x: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Tensor with shape (batch_size, num_classes)
        """
        # Process each timeframe
        processed_features = []
        
        for tf, features in x.items():
            # Input processing
            batch_size, seq_len, _ = features.shape
            features = features.reshape(-1, features.size(-1))
            features = self.input_layers[tf](features)
            features = features.reshape(batch_size, seq_len, -1)
            
            if self.use_transformer:
                # Transformer processing
                features = features.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
                features = self.pos_encoding(features)
                for block in self.transformer_blocks:
                    features = block(features)
                features = features.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
            else:
                # LSTM processing
                lstm_out, _ = self.lstm_layers[tf](features)
                
                if self.attention:
                    # Apply attention
                    lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
                    lstm_out = self.attention_layers[tf](lstm_out)
                    lstm_out = lstm_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
                
                features = lstm_out
            
            # Take the last timestep
            features = features[:, -1, :]
            processed_features.append(features)
        
        # Concatenate features from all timeframes
        combined_features = torch.cat(processed_features, dim=1)
        
        # Output processing
        output = self.output_layers(combined_features)
        return output

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series data with multiple timeframes
    """
    
    def __init__(self, 
                input_dims: Dict[str, int],
                hidden_dims: int = 128,
                num_heads: int = 4,
                num_layers: int = 2,
                dropout: float = 0.2,
                max_seq_len: int = 100):
        """
        Initialize the transformer model
        
        Parameters:
        - input_dims: Dictionary mapping timeframe names to input feature dimensions
        - hidden_dims: Dimension of hidden layers
        - num_heads: Number of attention heads
        - num_layers: Number of transformer layers
        - dropout: Dropout probability
        - max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.timeframes = list(input_dims.keys())
        self.hidden_dims = hidden_dims
        
        # Feature embedding layers for each timeframe
        self.feature_embeddings = nn.ModuleDict()
        for tf, dim in input_dims.items():
            self.feature_embeddings[tf] = nn.Linear(dim, hidden_dims)
        
        # Position encoding
        self.position_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dims)
        )
        
        # Transformer encoders for each timeframe
        self.transformers = nn.ModuleDict()
        for tf in self.timeframes:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dims,
                nhead=num_heads,
                dim_feedforward=hidden_dims * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformers[tf] = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )
        
        # Cross-timeframe attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dims * len(self.timeframes), hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims // 2)
        self.fc3 = nn.Linear(hidden_dims // 2, 3)  # 3 classes: buy, sell, hold
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position encoding
        self._init_position_encoding(max_seq_len, hidden_dims)
    
    def _init_position_encoding(self, max_seq_len: int, hidden_dims: int):
        """Initialize the position encoding using sine and cosine functions"""
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dims, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dims))
        
        pos_encoding = torch.zeros(1, max_seq_len, hidden_dims)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        self.position_encoding.data = pos_encoding
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Tensor with shape (batch_size, num_classes)
        """
        # Process each timeframe
        encoded_timeframes = {}
        for tf, transformer in self.transformers.items():
            if tf in inputs:
                # Apply feature embedding
                x = self.feature_embeddings[tf](inputs[tf])
                
                # Add position encoding
                seq_len = x.size(1)
                x = x + self.position_encoding[:, :seq_len, :]
                
                # Apply transformer
                mask = None  # Can add masking if needed
                x = transformer(x, mask)
                
                # Use the final token as the timeframe representation
                encoded_timeframes[tf] = x[:, -1, :]
            else:
                # Handle missing timeframe
                batch_size = next(iter(inputs.values())).size(0)
                device = next(iter(inputs.values())).device
                
                # Create a zero tensor of appropriate size
                encoded_timeframes[tf] = torch.zeros(
                    batch_size, 
                    self.hidden_dims,
                    device=device
                )
        
        # Apply cross-timeframe attention
        tf_tensors = []
        for tf in self.timeframes:
            # Get the encoded representation for this timeframe
            tf_encoding = encoded_timeframes[tf].unsqueeze(1)  # Add sequence dimension
            
            # Create a context vector from all other timeframes
            context_list = []
            for other_tf in self.timeframes:
                if other_tf != tf:
                    context_list.append(encoded_timeframes[other_tf].unsqueeze(1))
            
            if context_list:
                context = torch.cat(context_list, dim=1)
                
                # Apply cross-attention
                attended, _ = self.cross_attention(
                    query=tf_encoding,
                    key=context,
                    value=context
                )
                
                # Combine with original encoding
                combined = tf_encoding + attended
                tf_tensors.append(combined.squeeze(1))
            else:
                tf_tensors.append(tf_encoding.squeeze(1))
        
        # Concatenate all timeframe representations
        combined = torch.cat(tf_tensors, dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(self.dropout(combined)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        
        return x

class TimeSeriesForecaster(nn.Module):
    """
    Model for time series forecasting with multiple timeframes
    """
    
    def __init__(self, 
                input_dims: Dict[str, int],
                forecast_horizon: int = 5,
                hidden_dims: int = 128,
                num_layers: int = 2,
                dropout: float = 0.2):
        """
        Initialize the forecaster model
        
        Parameters:
        - input_dims: Dictionary mapping timeframe names to input feature dimensions
        - forecast_horizon: Number of steps to forecast
        - hidden_dims: Dimension of hidden layers
        - num_layers: Number of LSTM/GRU layers
        - dropout: Dropout probability
        """
        super().__init__()
        
        self.timeframes = list(input_dims.keys())
        self.hidden_dims = hidden_dims
        self.forecast_horizon = forecast_horizon
        
        # Encoder LSTMs for each timeframe
        self.encoders = nn.ModuleDict()
        for tf, dim in input_dims.items():
            self.encoders[tf] = nn.LSTM(
                input_size=dim,
                hidden_size=hidden_dims,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=1,  # Single value input (previous prediction)
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention for combining timeframes
        self.attention = nn.Linear(hidden_dims * len(self.timeframes), hidden_dims)
        
        # Output layer
        self.fc = nn.Linear(hidden_dims, 1)  # Predict single value (price)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Dict[str, torch.Tensor], targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        - targets: Optional target tensor for teacher forcing
                  Shape (batch_size, forecast_horizon)
        
        Returns:
        - Tensor with shape (batch_size, forecast_horizon)
        """
        batch_size = next(iter(inputs.values())).size(0)
        device = next(iter(inputs.values())).device
        
        # Process each timeframe through its encoder
        hidden_states = {}
        cell_states = {}
        
        for tf, encoder in self.encoders.items():
            if tf in inputs:
                # Run the encoder
                _, (hidden, cell) = encoder(inputs[tf])
                
                hidden_states[tf] = hidden
                cell_states[tf] = cell
            else:
                # Handle missing timeframe
                hidden_states[tf] = torch.zeros(
                    self.num_layers, batch_size, self.hidden_dims, device=device
                )
                cell_states[tf] = torch.zeros(
                    self.num_layers, batch_size, self.hidden_dims, device=device
                )
        
        # Combine hidden states from all timeframes using attention
        combined_hidden = []
        combined_cell = []
        
        for layer in range(self.num_layers):
            # Concatenate hidden states from all timeframes
            hidden_concat = torch.cat([hidden_states[tf][layer] for tf in self.timeframes], dim=1)
            
            # Apply attention to combine timeframes
            attended_hidden = torch.tanh(self.attention(hidden_concat))
            combined_hidden.append(attended_hidden)
            
            # Combine cell states (simple average)
            cell_concat = torch.stack([cell_states[tf][layer] for tf in self.timeframes], dim=0)
            combined_cell.append(torch.mean(cell_concat, dim=0))
        
        # Stack combined states
        decoder_hidden = torch.stack(combined_hidden, dim=0)
        decoder_cell = torch.stack(combined_cell, dim=0)
        
        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1, 1, device=device)
        
        # Autoregressive decoding
        forecasts = []
        
        for t in range(self.forecast_horizon):
            # Run decoder for one step
            output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # Predict next value
            forecast = self.fc(output.squeeze(1))
            forecasts.append(forecast)
            
            # Use target for next input if available (teacher forcing)
            if targets is not None and torch.rand(1).item() < 0.5:  # 50% teacher forcing
                decoder_input = targets[:, t].unsqueeze(1).unsqueeze(2)
            else:
                decoder_input = forecast.detach().unsqueeze(1)
        
        # Stack forecasts
        return torch.stack(forecasts, dim=1) 