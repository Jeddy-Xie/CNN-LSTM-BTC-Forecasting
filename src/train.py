import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime, timedelta
import argparse

from data.data_collector import CryptoDataCollector
from models.cnn_lstm import CNNLSTM
from utils.logger import setup_logger

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    logger,
    save_dir
):
    """
    Train the model.
    
    Args:
        model: The CNN-LSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on (cuda/cpu)
        logger: Logger instance
        save_dir: Directory to save model checkpoints
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log progress
        logger.info(f'Epoch: {epoch+1}/{num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'best_model.pth')
            )

def main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--sequence_length', type=int, default=60)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('training')
    save_dir = os.path.join('models', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Data collection and preprocessing
    collector = CryptoDataCollector()
    df = collector.fetch_ohlcv(
        start_date=datetime.now() - timedelta(days=365*2)
    )
    X_train, X_test, y_train, y_test = collector.preprocess_data(
        df,
        sequence_length=args.sequence_length
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Initialize model, criterion, and optimizer
    model = CNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.num_epochs,
        device,
        logger,
        save_dir
    )

if __name__ == '__main__':
    main() 