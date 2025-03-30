import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_classes=1, dropout=0.25):
        super(CNNLSTM, self).__init__()
        
        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU()
        )
        cnn_output_dim = 16 * (input_dim)
        assert cnn_output_dim == 80, "CNN output dimension is not correct"

        # LSTM part
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        if x.dim() == 3:
            # Add a dimension at the end so each time step becomes a 5x1 "image"
            x = x.unsqueeze(-1)  # Now shape: (B, T, 5, 1)
        batch_size, seq_len, H, W = x.size()  # Here H = 5, W = 1
        
        # Combine batch and sequence dimensions so the CNN processes each time step individually
        x = x.view(batch_size * seq_len, 1, H, W)  # Now shape: (B*T, 1, 5, 1)
        
        # Pass through the CNN
        x = self.cnn(x)  # Expected shape: (B*T, 16, 5, 1)
        
        # Flatten the CNN output per time step (16 * 5 = 80 features)
        x = x.view(batch_size, seq_len, -1)  # Shape: (B, T, 80)
        
        # Pass the sequence of feature vectors through the LSTM
        x_out, _ = self.lstm(x)  # Shape: (B, T, hidden_dim)
        
        # Use the output of the last time step for prediction
        x_last = x_out[:, -1, :]  # Shape: (B, hidden_dim)
        
        # Final prediction through the fully connected layer
        output = self.fc(x_last)  # Shape: (B, num_classes)
        
        return output

