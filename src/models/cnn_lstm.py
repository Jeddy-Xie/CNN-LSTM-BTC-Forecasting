import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for Bitcoin price movement prediction.
    
    Architecture:
    1. CNN layers for spatial feature extraction from OHLCV data
    2. LSTM layers for temporal pattern learning
    3. Fully connected layers for final prediction
    """
    
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_classes=2):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_dim (int): Number of input features (default: 5 for OHLCV)
            hidden_dim (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (2 for binary up/down prediction)
        """
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Model predictions
        """
        # Reshape for CNN (batch_size, input_dim, sequence_length)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Reshape for LSTM (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM sequence processing
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output for prediction
        x = lstm_out[:, -1, :]
        
        # Final prediction
        x = self.fc(x)
        return x 