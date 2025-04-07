import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class MultiTimeframeModel(nn.Module):
    """
    Deep learning model that processes multiple timeframes
    
    Features:
    - Separate input paths for each timeframe
    - LSTM/GRU layers for sequence processing
    - Attention mechanisms between timeframes
    - Dense output layers for predictions
    """
    
    def __init__(self, 
                input_dims: Dict[str, int], 
                hidden_dims: int = 128, 
                num_layers: int = 2,
                dropout: float = 0.2,
                bidirectional: bool = True,
                attention: bool = True,
                num_classes: int = 3):
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
        """
        super().__init__()
        
        self.timeframes = list(input_dims.keys())
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_directions = 2 if bidirectional else 1
        self.num_classes = num_classes
        
        # Create encoder LSTMs for each timeframe
        self.encoders = nn.ModuleDict()
        for tf, dim in input_dims.items():
            self.encoders[tf] = nn.LSTM(
                input_size=dim,
                hidden_size=hidden_dims,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Attention mechanism
        if attention:
            attn_dim = hidden_dims * self.num_directions
            self.attn_weights = nn.ParameterDict()
            
            # Create attention weights for each timeframe
            for tf in self.timeframes:
                self.attn_weights[tf] = nn.Parameter(torch.randn(attn_dim, attn_dim))
                
            self.attn_combine = nn.Linear(len(self.timeframes) * attn_dim, attn_dim)
        
        # Output layers
        output_dim = hidden_dims * self.num_directions * (1 if attention else len(self.timeframes))
        
        self.fc1 = nn.Linear(output_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims // 2)
        self.fc3 = nn.Linear(hidden_dims // 2, self.num_classes)  # Classes: buy, sell, hold
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Tensor with shape (batch_size, num_classes)
        """
        # Process each timeframe through its encoder
        encoded_timeframes = {}
        for tf, encoder in self.encoders.items():
            if tf in inputs:
                # Run the encoder
                output, (hidden, cell) = encoder(inputs[tf])
                
                # Get the final hidden state(s)
                if self.bidirectional:
                    # Fix: Check hidden dimensions and reshape correctly
                    # hidden shape is (num_layers * num_directions, batch_size, hidden_size)
                    batch_size = hidden.size(1)
                    
                    # Reshape to get the last layer from both directions
                    if hidden.size(0) >= 2:  # we have at least 2 hidden states (forward and backward)
                        # Last layer, forward direction
                        forward = hidden[-2, :, :]
                        # Last layer, backward direction
                        backward = hidden[-1, :, :]
                        
                        # Concatenate along the feature dimension
                        final_hidden = torch.cat([forward, backward], dim=1)
                    else:
                        # If for some reason we only have one direction, just use it
                        final_hidden = hidden.view(batch_size, -1)
                else:
                    # For unidirectional, just take the last layer's hidden state
                    final_hidden = hidden[-1]
                
                encoded_timeframes[tf] = final_hidden
            else:
                # Handle missing timeframe
                batch_size = next(iter(inputs.values())).size(0)
                device = next(iter(inputs.values())).device
                
                # Create a zero tensor of appropriate size
                encoded_timeframes[tf] = torch.zeros(
                    batch_size, 
                    self.hidden_dims * self.num_directions,
                    device=device
                )
        
        # Apply attention mechanism if enabled
        if self.attention:
            # Calculate attention scores between timeframes
            attn_applied = []
            
            for tf1 in self.timeframes:
                # Apply attention between current timeframe and all others
                attended_vector = encoded_timeframes[tf1]
                
                for tf2 in self.timeframes:
                    if tf1 != tf2:
                        # Calculate attention score
                        attn_weight = self.attn_weights[tf1]
                        
                        # Fix: Reshape the hidden state for correct matrix multiplication
                        # encoded_timeframes[tf1/2] is of shape [batch_size, hidden_dim]
                        
                        # Calculate similarity score
                        attn_projection = torch.matmul(encoded_timeframes[tf1], attn_weight)  # [batch_size, hidden_dim]
                        attn_score = torch.bmm(
                            attn_projection.unsqueeze(1),                 # [batch_size, 1, hidden_dim]
                            encoded_timeframes[tf2].unsqueeze(2)          # [batch_size, hidden_dim, 1]
                        ).squeeze(2)  # [batch_size, 1]
                        
                        # Apply attention score with softmax
                        # Note: Since attn_score is [batch_size, 1], we don't need dim parameter for softmax
                        # Adding a small epsilon to avoid numerical instability
                        attn_score = torch.sigmoid(attn_score + 1e-6)
                        
                        # Scale the tf2 encoding by attention score
                        attended = attn_score.unsqueeze(1) * encoded_timeframes[tf2].unsqueeze(1)  # [batch_size, 1, hidden_dim]
                        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
                        
                        # Update the attended vector
                        attended_vector = attended_vector + attended
                
                attn_applied.append(attended_vector)
            
            # Combine attended vectors
            combined = torch.cat(attn_applied, dim=1)
            combined = self.attn_combine(combined)
        
        else:
            # Simple concatenation of all timeframe encodings
            combined = torch.cat([encoded_timeframes[tf] for tf in self.timeframes], dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(self.dropout(combined)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        
        return x

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