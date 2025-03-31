import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch.nn as nn

from src.models.cnn_lstm import CNNLSTM  

from comet_ml import Experiment

# --- Data Loading ---
def data_load(filePath):
    """
    Load and preprocess data.
    
    Args:
        filePath (str): Path to the CSV file.
    
    Returns:
        data (pd.DataFrame): Preprocessed DataFrame.
    """
    # Load CSV
    data = pd.read_csv(filePath)

    # Drop date column if exists
    if 'date' in data.columns:
        data.drop(columns=['date'], inplace=True)

    # Convert timestamp to datetime and set as index
    data['date'] = pd.to_datetime(data['unix'], unit='s')
    data.set_index('date', inplace=True)

    # Drop symbol column if exists
    if 'symbol' in data.columns:
        data.drop(columns=['symbol'], inplace=True)

    # Rename columns so that:
    # - 'close' becomes 'y' (target)
    # - 'Volume BTC' becomes 'x1'
    # - 'Volume USD' becomes 'x2'
    # - 'open' becomes 'x3'
    # - 'high' becomes 'x4'
    # - 'low' becomes 'x5'
    data.rename(columns={
        'close': 'y', 
        'Volume BTC': 'x1', 
        'Volume USD': 'x2', 
        'open': 'x3', 
        'high': 'x4', 
        'low': 'x5'
    }, inplace=True)
    data.sort_index(inplace=True)
    return data

# --- Data Scaling ---
def scale_data(data, scaler_type='minmax'):
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif scaler_type.lower() == 'robust':
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}. Use 'minmax', 'standard', or 'robust'")
    return scaler.fit_transform(data)

# --- Create Time Series Tensors ---
def create_time_series_tensors(data, sequence_length=10, target_index=5):
    """
    Create sequences (X) and targets (y) for time series forecasting.
    
    Args:
        data (np.array): Scaled data of shape (num_total_steps, num_features).
        sequence_length (int): Number of time steps per sample.
        target_index (int): Column index for the target.
        
    Returns:
        X_tensor (torch.Tensor): Tensor of shape (num_samples, sequence_length, num_features).
        y_tensor (torch.Tensor): Tensor of shape (num_samples, 1).
    """
    input_data = np.delete(data, target_index, axis=1)
    X_list, y_list = [], []
    for i in range(len(data) - sequence_length):
        input_sequence = input_data[i:i+sequence_length]
        X_list.append(input_sequence)
        y_list.append(data[i+sequence_length, target_index])
    X = np.array(X_list)
    y = np.array(y_list)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X_tensor, y_tensor

# --- Training Function ---
def train(train_loader, learning_rate, num_epochs, hidden_dim, num_layers, dropout):
    model = CNNLSTM(input_dim=5, hidden_dim=int(hidden_dim), num_layers=num_layers,
                    dropout=dropout, num_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()  # Mean Absolute Error Loss
    
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# --- Evaluation Function ---
def evaluate(model, test_loader, criterion=nn.L1Loss()):
    device = next(model.parameters()).device  # Use the same device as the model
    model.eval()
    test_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            batch_size = batch_X.size(0)
            test_loss += loss.item() * batch_size
            total_samples += batch_size
    test_loss /= total_samples
    return -test_loss  # Negative loss for maximization

# --- Objective Function for Bayesian Optimization ---
def objective(data, learning_rate, batch_size, num_epochs, hidden_dim, num_layers, dropout, sequence_length, scaler_type_num, experiment):
    # Create a new Comet experiment for this iteration 
    # Convert hyperparameters to proper types
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    num_layers = int(num_layers)
    sequence_length = int(sequence_length)
    
    # Map scaler_type_num (a continuous value) to a categorical scaler option.
    scaler_options = ['minmax', 'standard', 'robust']
    scaler_type = scaler_options[int(round(scaler_type_num))]
    
    # Log hyperparameters to Comet
    experiment.log_parameters({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "hidden_dim": int(hidden_dim),
        "num_layers": num_layers,
        "dropout": dropout,
        "sequence_length": sequence_length,
        "scaler_type": scaler_type,
    })
    
    # Prepare the data
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    data_array = data[features].values
    
    # Scale the data
    scaled_data = scale_data(data_array, scaler_type)
    
    # Create sequences (time series) and corresponding targets.
    close_index = features.index('y')
    X_tensor, y_tensor = create_time_series_tensors(scaled_data, sequence_length, target_index=close_index)
    
    # Train-test split (80/20)
    train_size = int(0.8 * len(X_tensor))
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_test = X_tensor[train_size:]
    y_test = y_tensor[train_size:]
    
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train the model and evaluate
    model = train(train_loader, learning_rate, num_epochs, hidden_dim, num_layers, dropout)
    test_loss = evaluate(model, test_loader)
    
    # Log the metric (test loss) and end the experiment
    experiment.log_metric("test_loss", test_loss)
    return test_loss