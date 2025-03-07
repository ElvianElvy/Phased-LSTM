import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.phased_lstm import CryptoPhasedLSTM
from utils.metrics import calculate_metrics

def train_model(symbol: str, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
               hidden_size: int = 128, num_layers: int = 2, sequence_length: int = 30,
               train_days: int = 365, validation_split: float = 0.2, save_dir: str = "saved_models"):
    """
    Train the Phased LSTM model on cryptocurrency data.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_size: Hidden size of the LSTM
        num_layers: Number of LSTM layers
        sequence_length: Number of timesteps in each input sequence
        train_days: Number of days of historical data to use for training
        validation_split: Fraction of data to use for validation
        save_dir: Directory to save the trained model
    
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    api = BinanceAPI()
    preprocessor = CryptoDataPreprocessor(sequence_length=sequence_length)
    
    print(f"Fetching historical data for {symbol}...")
    df = api.get_training_data(symbol=symbol, days=train_days)
    
    print(f"Processing data...")
    df = preprocessor.process_raw_data(df)
    
    print(f"Preparing sequences...")
    X, T, y = preprocessor.prepare_data(df)
    
    # Split into training and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    T_train, T_val = T[:split_idx], T[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, T_train, y_train)
    val_dataset = TensorDataset(X_val, T_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X.shape[2]  # Number of features
    model = CryptoPhasedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, T_batch, y_batch in progress_bar:
            # Move tensors to device
            X_batch, T_batch, y_batch = X_batch.to(device), T_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch, T_batch)
            loss = criterion(outputs, y_batch)
            
            # Add L2 regularization if implemented
            if hasattr(model, 'get_l2_regularization_loss'):
                reg_loss = model.get_l2_regularization_loss()
                loss = loss + reg_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, T_batch, y_batch in val_loader:
                # Move tensors to device
                X_batch, T_batch, y_batch = X_batch.to(device), T_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch, T_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Calculate validation metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Convert predictions back to original scale
        original_predictions = preprocessor.inverse_transform_predictions(all_predictions)
        original_targets = preprocessor.inverse_transform_predictions(all_targets)
        
        metrics = calculate_metrics(original_predictions, original_targets)
        history['val_metrics'].append(metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Val MAE: {metrics['mae']:.2f}, Val RMSE: {metrics['rmse']:.2f}, Val MAPE: {metrics['mape']:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"{symbol}_phased_lstm_{timestamp}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length,
                    'input_size': input_size
                }
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Save preprocessor
            preprocessor_path = os.path.join(save_dir, f"{symbol}_preprocessor_{timestamp}.pkl")
            import pickle
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            print(f"Preprocessor saved to {preprocessor_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot([m['mae'] for m in history['val_metrics']], label='MAE')
    plt.plot([m['rmse'] for m in history['val_metrics']], label='RMSE')
    plt.title('Validation Error Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot([m['mape'] for m in history['val_metrics']], label='MAPE (%)')
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    
    plt.subplot(2, 2, 4)
    plt.plot([m['dir_acc'] for m in history['val_metrics']], label='Directional Accuracy (%)')
    plt.title('Directional Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f"{symbol}_training_history_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return model, history, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--train_days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    
    args = parser.parse_args()
    
    train_model(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.seq_length,
        train_days=args.train_days,
        save_dir=args.save_dir
    )