import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#assert torch.cuda.is_available(), 'cuda is not available'
#assert device == 'cuda', 'device is not cuda'

# Set up the data directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
data_dir = os.path.join(current_dir, 'data', 'processed',)

# Import Handwritten Functions
from src.models.cnn_lstm import CNNLSTM
from src.train import *

# Import Bayesian Optimization
from bayes_opt import BayesianOptimization

btc = data_load(os.path.join(data_dir, 'BTC-USD.csv'))

# Hyperparameter Tuning
# --- Define the hyperparameter bounds ---
pbounds = {
    'learning_rate': (1e-5, 1e-1),
    'batch_size': (32, 128),
    'num_epochs': (1, 50),
    'hidden_dim': (16, 256),
    'num_layers': (1, 3),
    'dropout': (0.1, 0.5),
    'sequence_length': (5, 60),
    'scaler_type_num': (0, 2)  # Here 0 corresponds to 'minmax', 1 to 'standard', 2 to 'robust'
}

# --- Set up Bayesian Optimization ---

optimizer_bo = BayesianOptimization(
    f=lambda learning_rate, batch_size, num_epochs, hidden_dim, num_layers, dropout, sequence_length, scaler_type_num:
        objective(btc, learning_rate, batch_size, num_epochs, hidden_dim, num_layers, dropout, sequence_length, scaler_type_num),
    pbounds=pbounds,
    random_state=42,
)

# --- Run the optimization ---

optimizer_bo.maximize(
    init_points=10,
    n_iter=50,
)

# --- Print the best results ---
best_hyperparams = optimizer_bo.max
print("Best hyperparameters found:")
print(best_hyperparams)