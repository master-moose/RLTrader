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
                num_classes: int = 3,
                use_batch_norm: bool = True,
                use_residual: bool = True,
                embedding_dim: Optional[int] = None):
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
        - use_residual: Whether to use residual connections
        - embedding_dim: Optional dimension for initial feature embedding
        """
        super().__init__()
        
        self.timeframes = list(input_dims.keys())
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_directions = 2 if bidirectional else 1
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Optional feature embedding layers
        self.embeddings = nn.ModuleDict()
        if embedding_dim is not None:
            for tf, dim in input_dims.items():
                self.embeddings[tf] = nn.Sequential(
                    nn.Linear(dim, embedding_dim),
                    nn.BatchNorm1d(embedding_dim) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU()
                )
            # Update dimensions for the encoders
            encoder_input_dims = {tf: embedding_dim for tf in input_dims}
        else:
            encoder_input_dims = input_dims
        
        # Create encoder LSTMs for each timeframe
        self.encoders = nn.ModuleDict()
        for tf, dim in encoder_input_dims.items():
            self.encoders[tf] = nn.LSTM(
                input_size=dim,
                hidden_size=hidden_dims,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Batch normalization layers for encoder outputs
        if use_batch_norm:
            self.bn_encoders = nn.ModuleDict()
            for tf in self.timeframes:
                self.bn_encoders[tf] = nn.BatchNorm1d(hidden_dims * self.num_directions)
        
        # Improved attention mechanism with scaled dot-product attention
        if attention:
            attn_dim = hidden_dims * self.num_directions
            self.query_projections = nn.ModuleDict()
            self.key_projections = nn.ModuleDict()
            self.value_projections = nn.ModuleDict()
            
            for tf in self.timeframes:
                self.query_projections[tf] = nn.Linear(attn_dim, attn_dim)
                self.key_projections[tf] = nn.Linear(attn_dim, attn_dim)
                self.value_projections[tf] = nn.Linear(attn_dim, attn_dim)
            
            self.attn_combine = nn.Sequential(
                nn.Linear(len(self.timeframes) * attn_dim, attn_dim),
                nn.BatchNorm1d(attn_dim) if use_batch_norm else nn.Identity(),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
        
        # Output layers with batch normalization and residual connections
        output_dim = hidden_dims * self.num_directions * (1 if attention else len(self.timeframes))
        
        self.fc_layers = nn.ModuleList()
        
        # First layer 
        self.fc1 = nn.Sequential(
            nn.Linear(output_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Second layer with residual connection
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.BatchNorm1d(hidden_dims // 2) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection adapter if using residual connections
        if use_residual:
            self.residual_adapter = nn.Linear(hidden_dims, hidden_dims // 2)
        
        # Final layer
        self.fc3 = nn.Linear(hidden_dims // 2, self.num_classes)
        
        # Class balancing weights - initialized to equal weights
        self.class_weights = nn.Parameter(torch.ones(num_classes))
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Tensor with shape (batch_size, num_classes)
        """
        features = self.extract_features(inputs)
        
        # Pass through fully connected layers with residual connection
        x = self.fc1(features)
        
        # Apply residual connection
        if self.use_residual and hasattr(self, 'residual_adapter'):
            residual = self.residual_adapter(x)
            x = self.fc2(x) + residual
        else:
            x = self.fc2(x)
        
        # Final layer with class weights
        x = self.fc3(x)
        
        # Apply class weights for balanced learning
        x = x * self.class_weights
        
        return x
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from multiple timeframes without final classification
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Feature tensor with shape (batch_size, feature_dim)
        """
        # Preprocess inputs
        processed_inputs = self._preprocess_inputs(inputs)
        
        # Process each timeframe through its encoder
        encoded_timeframes = {}
        for tf in self.timeframes:
            if tf in processed_inputs:
                # Apply optional embedding
                if tf in self.embeddings:
                    # Reshape for batch norm: [batch, seq, features] -> [batch*seq, features]
                    batch_size, seq_len, input_dim = processed_inputs[tf].shape
                    flat_input = processed_inputs[tf].reshape(-1, input_dim)
                    
                    # Apply embedding
                    embedded = self.embeddings[tf](flat_input)
                    
                    # Reshape back: [batch*seq, embedding_dim] -> [batch, seq, embedding_dim]
                    processed_inputs[tf] = embedded.view(batch_size, seq_len, -1)
                
                # Run the encoder
                output, (hidden, _) = self.encoders[tf](processed_inputs[tf])
                
                # Get the final hidden state(s)
                if self.bidirectional:
                    # Reshape hidden for easy access to directions
                    # hidden shape: [num_layers * num_directions, batch, hidden]
                    batch_size = hidden.size(1)
                    num_layers = hidden.size(0) // self.num_directions
                    
                    # Extract last layer's hidden states (forward and backward)
                    hidden_forward = hidden[num_layers*2-2, :, :]  # Last forward direction
                    hidden_backward = hidden[num_layers*2-1, :, :]  # Last backward direction
                    
                    # Concatenate forward and backward states
                    final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
                else:
                    # For unidirectional, just take the last layer's hidden state
                    final_hidden = hidden[-1]
                
                # Apply batch normalization if enabled
                if self.use_batch_norm and tf in self.bn_encoders:
                    final_hidden = self.bn_encoders[tf](final_hidden)
                
                encoded_timeframes[tf] = final_hidden
            else:
                # Handle missing timeframe with zero tensor
                batch_size = next(iter(processed_inputs.values())).size(0)
                device = next(iter(processed_inputs.values())).device
                
                encoded_timeframes[tf] = torch.zeros(
                    batch_size, 
                    self.hidden_dims * self.num_directions,
                    device=device
                )
        
        # Apply attention mechanism if enabled
        if self.attention:
            combined = self._apply_attention(encoded_timeframes)
        else:
            # Simple concatenation of all timeframe encodings
            combined = torch.cat([encoded_timeframes[tf] for tf in self.timeframes], dim=1)
        
        return combined
    
    def _preprocess_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess input tensors before feeding to encoders
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
        
        Returns:
        - Processed inputs
        """
        processed = {}
        
        # Process each timeframe input
        for tf, tensor in inputs.items():
            if tf in self.timeframes:
                # Handle potential dimension mismatches
                if tensor.dim() == 2:
                    # Add sequence dimension if missing: [batch, features] -> [batch, 1, features]
                    tensor = tensor.unsqueeze(1)
                
                # Apply input normalization to stabilize training
                # Compute mean and std across batch and time dimensions
                mean = tensor.mean(dim=[0, 1], keepdim=True)
                std = tensor.std(dim=[0, 1], keepdim=True) + 1e-6  # Avoid division by zero
                normalized = (tensor - mean) / std
                
                processed[tf] = normalized
        
        return processed
    
    def _apply_attention(self, encoded_timeframes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply scaled dot-product attention mechanism between timeframes
        
        Parameters:
        - encoded_timeframes: Dictionary of encoded features for each timeframe
        
        Returns:
        - Combined attention-weighted features
        """
        attention_outputs = []
        
        # Use scaled dot-product attention
        for tf_query in self.timeframes:
            # Project query
            query = self.query_projections[tf_query](encoded_timeframes[tf_query])
            
            # Attend to all other timeframes
            attended_values = []
            
            for tf_key in self.timeframes:
                # Project key and value
                key = self.key_projections[tf_key](encoded_timeframes[tf_key])
                value = self.value_projections[tf_key](encoded_timeframes[tf_key])
                
                # Calculate attention scores
                attention_scores = torch.matmul(query.unsqueeze(1), key.unsqueeze(2)) / (self.hidden_dims ** 0.5)
                attention_scores = torch.sigmoid(attention_scores).squeeze(2)
                
                # Apply attention scores to values
                attended = attention_scores.unsqueeze(1) * value.unsqueeze(1)
                attended = attended.squeeze(1)
                
                attended_values.append(attended)
            
            # Sum all attended values
            if attended_values:
                summed_attended = sum(attended_values)
                attention_outputs.append(summed_attended)
        
        # Concatenate all attention outputs
        concatenated = torch.cat(attention_outputs, dim=1)
        
        # Combine attention outputs
        combined = self.attn_combine(concatenated)
        
        return combined
    
    def predict_probabilities(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass that returns class probabilities
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
        
        Returns:
        - Tensor with shape (batch_size, num_classes) containing probabilities
        """
        logits = self.forward(inputs)
        return F.softmax(logits, dim=1)
    
    def predict_action(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Predict the best action (for use with RL agents)
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
        
        Returns:
        - Action index (0=hold, 1=buy, 2=sell)
        """
        with torch.no_grad():
            logits = self.forward(inputs)
            probabilities = F.softmax(logits, dim=1)
            return torch.argmax(probabilities, dim=1).item()

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