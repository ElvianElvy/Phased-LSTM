import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhasedLSTMCell(nn.Module):
    """
    Phased LSTM Cell implementation in PyTorch.
    
    Phased LSTM extends the LSTM architecture with a time gate controlled by
    an oscillation, which helps with learning long-term dependencies and dealing
    with irregularly sampled data.
    
    Based on the paper: "Phased LSTM: Accelerating Recurrent Network Training for Long or 
    Event-based Sequences" by Daniel Neil, Michael Pfeiffer, Shih-Chii Liu
    https://arxiv.org/abs/1610.09513
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(PhasedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM parameters
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias)
        
        # Time gate parameters
        self.time_gate = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        
        # Initialize time gate parameters (τ, r, s)
        # τ (period): initialized to be a bit longer than the average sequence
        # r (shift): initialized randomly between 0 and period
        # s (on-ratio): initialized to a small value (e.g., 0.05)
        self.tau = nn.Parameter(torch.Tensor(hidden_size))
        self.r = nn.Parameter(torch.Tensor(hidden_size))
        self.s = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize time gate parameters
        # τ (period) is initialized to a random value between 1 and 1000
        nn.init.uniform_(self.tau, 1, 1000)
        # r (shift) is initialized to a random value between 0 and 1
        nn.init.uniform_(self.r, 0, 1)
        # s (on-ratio) is initialized to a small value
        nn.init.constant_(self.s, 0.05)
    
    def forward(self, input, time, hidden=None):
        """
        Forward pass for the Phased LSTM cell.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            time: Time tensor of shape (batch_size, 1)
            hidden: Tuple of (h, c) where h and c are tensors of shape (batch_size, hidden_size)
                   If None, initialized with zeros
        
        Returns:
            h_new, c_new: New hidden and cell states
        """
        batch_size = input.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=input.device)
            c = torch.zeros(batch_size, self.hidden_size, device=input.device)
        else:
            h, c = hidden
        
        # Regular LSTM cell computation
        h_lstm, c_lstm = self.lstm_cell(input, (h, c))
        
        # Time gate computation
        # Compute the phase Φ(t) based on the time input
        phi = (((time - self.r) % self.tau) / self.tau).expand_as(h)
        
        # Compute the time gate k_t
        k = torch.zeros_like(phi)
        
        # First part: 0 <= phi < s/2
        mask1 = (phi < self.s / 2).float()
        k += mask1 * 2 * phi / self.s
        
        # Second part: s/2 <= phi < s
        mask2 = ((phi >= self.s / 2) & (phi < self.s)).float()
        k += mask2 * (2 - 2 * phi / self.s)
        
        # Apply the time gate
        h_new = k * h_lstm + (1 - k) * h
        c_new = k * c_lstm + (1 - k) * c
        
        return h_new, c_new


class PhasedLSTM(nn.Module):
    """
    Phased LSTM layer that uses PhasedLSTMCell for each step in the sequence.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0):
        super(PhasedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Create a list of Phased LSTM cells
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cell_list.append(PhasedLSTMCell(layer_input_size, hidden_size, bias))
    
    def forward(self, input, time, hidden=None):
        """
        Forward pass for the Phased LSTM.
        
        Args:
            input: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True
                  else (seq_len, batch_size, input_size)
            time: Time tensor of shape (batch_size, seq_len, 1) if batch_first=True
                  else (seq_len, batch_size, 1)
            hidden: Tuple of (h_0, c_0) where h_0 and c_0 are tensors of shape 
                   (num_layers, batch_size, hidden_size)
                   If None, initialized with zeros
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_size) if batch_first=True
                   else (seq_len, batch_size, hidden_size)
            h_n, c_n: Hidden and cell states for the last step, each of shape 
                     (num_layers, batch_size, hidden_size)
        """
        # Adjust dimensions if not batch_first
        if not self.batch_first:
            input = input.transpose(0, 1)
            time = time.transpose(0, 1)
        
        batch_size = input.size(0)
        seq_len = input.size(1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
        else:
            h, c = hidden
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        output = []
        
        # Process each time step
        for t in range(seq_len):
            input_t = input[:, t, :]
            time_t = time[:, t, :]
            
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.cell_list[layer](input_t, time_t, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.cell_list[layer](h[layer-1], time_t, (h[layer], c[layer]))
                
                # Apply dropout except for the last layer
                if layer < self.num_layers - 1 and self.dropout > 0:
                    h[layer] = F.dropout(h[layer], p=self.dropout, training=self.training)
            
            output.append(h[-1])
        
        # Stack outputs
        output = torch.stack(output, dim=1)
        
        # Stack h and c for return
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        # Adjust dimensions if not batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h_n, c_n)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important timesteps.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention layers
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.projection = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """
        Apply temporal attention to the sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Compute query, key, value projections
        q = self.query(x)  # (batch_size, seq_len, hidden_size)
        k = self.key(x)    # (batch_size, seq_len, hidden_size)
        v = self.value(x)  # (batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(hidden_size)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        context = torch.bmm(attn_weights, v)  # (batch_size, seq_len, hidden_size)
        
        # Apply output projection
        output = self.projection(context)
        
        # Apply residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output


class CryptoPhasedLSTM(nn.Module):
    """
    Enhanced cryptocurrency price prediction model using Phased LSTM.
    Includes attention mechanisms, residual connections, and regularization.
    Predicts daily open and close prices for the next week.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, l2_reg=1e-5):
        super(CryptoPhasedLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Phased LSTM layers
        self.phased_lstm = PhasedLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_size)
        
        # Feature extraction layers with residual connections
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(2)
        ])
        
        # Output layers with gradually decreasing dimensions
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),  # Less dropout in final layers
            nn.Linear(hidden_size // 2, 14)  # 7 days * 2 values (open, close)
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'lstm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                elif len(param.shape) == 1:
                    # For 1D parameters like bias, initialize with zeros
                    nn.init.zeros_(param)
    
    def forward(self, x, time):
        """
        Forward pass with enhanced processing.
        
        Args:
            x: Input features of shape (batch_size, seq_len, input_size)
            time: Time tensor of shape (batch_size, seq_len, 1)
        
        Returns:
            predictions: Predicted prices of shape (batch_size, 14)
                        representing open and close prices for 7 days
        """
        # Apply input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Run through Phased LSTM
        output, _ = self.phased_lstm(x, time)
        
        # Apply temporal attention
        attended_output = self.temporal_attention(output)
        
        # Take the last timestep output
        last_output = attended_output[:, -1, :]
        
        # Apply feature extraction with residual connections
        feature_output = last_output
        for layer in self.feature_layers:
            residual = feature_output
            feature_output = layer(feature_output)
            feature_output = feature_output + residual  # Residual connection
        
        # Apply output layers
        predictions = self.output_layers(feature_output)
        
        return predictions
    
    def get_l2_regularization_loss(self):
        """Calculate L2 regularization loss for the model parameters"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss