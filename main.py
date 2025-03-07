import os
import argparse
import json
import pickle
import time
from datetime import datetime

import torch
import pandas as pd

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.phased_lstm import CryptoPhasedLSTM
from train import train_model
from predict import predict_crypto_prices
from config import load_config


def list_available_cryptos():
    """List available cryptocurrencies on Binance."""
    print("Fetching available cryptocurrencies from Binance...")
    api = BinanceAPI()
    symbols = api.get_available_symbols()
    
    # Filter for common quote assets
    quote_assets = ["USDT", "BUSD", "BTC", "ETH"]
    filtered_symbols = []
    
    for symbol in symbols:
        for quote in quote_assets:
            if symbol.endswith(quote) and not symbol.startswith("USDT"):
                filtered_symbols.append(symbol)
                break
    
    # Group by quote asset
    grouped = {}
    for symbol in filtered_symbols:
        for quote in quote_assets:
            if symbol.endswith(quote):
                if quote not in grouped:
                    grouped[quote] = []
                grouped[quote].append(symbol)
                break
    
    # Print in organized format
    print(f"\nAvailable cryptocurrencies on Binance ({len(filtered_symbols)} pairs):")
    for quote, symbols in grouped.items():
        print(f"\n{quote} pairs ({len(symbols)}):")
        # Print in multiple columns
        col_width = 12
        cols = 6
        symbols_sorted = sorted(symbols)
        for i in range(0, len(symbols_sorted), cols):
            row = symbols_sorted[i:i+cols]
            print("  ".join(symbol.ljust(col_width) for symbol in row))


def find_latest_model(symbol, model_dir="saved_models"):
    """Find the latest trained model for a symbol."""
    if not os.path.exists(model_dir):
        return None, None
    
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Filter for model files matching the symbol
    model_files = [f for f in files if f.startswith(f"{symbol}_phased_lstm_") and f.endswith(".pt")]
    preprocessor_files = [f for f in files if f.startswith(f"{symbol}_preprocessor_") and f.endswith(".pkl")]
    
    if not model_files or not preprocessor_files:
        return None, None
    
    # Sort by timestamp in filename
    model_files.sort(reverse=True)
    preprocessor_files.sort(reverse=True)
    
    # Return paths
    model_path = os.path.join(model_dir, model_files[0])
    preprocessor_path = os.path.join(model_dir, preprocessor_files[0])
    
    return model_path, preprocessor_path


def train_new_model(args, config):
    """Train a new model with the specified configuration."""
    print(f"\n=== Training new model for {args.symbol} ===\n")
    
    # Extract training parameters from config
    train_params = config["training"]
    
    # Train the model
    model, history, preprocessor = train_model(
        symbol=args.symbol,
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        learning_rate=train_params["learning_rate"],
        hidden_size=train_params["hidden_size"],
        num_layers=train_params["num_layers"],
        sequence_length=train_params["sequence_length"],
        train_days=train_params["train_days"],
        validation_split=train_params["validation_split"],
        save_dir=args.model_dir
    )
    
    print("\nTraining completed.")
    
    # Find paths to the saved model and preprocessor
    model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    if model_path and preprocessor_path:
        return model_path, preprocessor_path
    else:
        raise FileNotFoundError("Could not find the trained model files.")


def predict_prices(args, config, model_path=None, preprocessor_path=None):
    """Predict prices for the specified cryptocurrency."""
    print(f"\n=== Predicting prices for {args.symbol} ===\n")
    
    # If paths are not provided, find the latest model
    if not model_path or not preprocessor_path:
        model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    # Check if model exists
    if not model_path or not preprocessor_path:
        print(f"No trained model found for {args.symbol}. Please train a model first.")
        return
    
    # Extract prediction parameters from config
    pred_params = config["prediction"]
    
    # Make predictions
    df_pred, summary = predict_crypto_prices(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        symbol=args.symbol,
        visualize=pred_params["create_visualization"],
        output_dir=args.output_dir
    )
    
    return df_pred, summary


def interactive_mode():
    """Run the application in interactive mode."""
    # Load configuration
    config = load_config()
    
    # Initialize Binance API
    api = BinanceAPI()
    
    print("=== Crypto Price Prediction with Phased LSTM ===\n")
    
    while True:
        print("\nOptions:")
        print("1. List available cryptocurrencies")
        print("2. Train a new model")
        print("3. Predict prices")
        print("4. Train and predict")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            list_available_cryptos()
        
        elif choice == "2" or choice == "3" or choice == "4":
            # Get symbol
            symbol = input("\nEnter cryptocurrency symbol (e.g., BTCUSDT): ").upper()
            
            # Create parser for argument handling
            parser = argparse.ArgumentParser()
            parser.add_argument("--symbol", type=str, default=symbol)
            parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"])
            parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"])
            args = parser.parse_args([])
            
            if choice == "2":
                # Train a new model
                try:
                    model_path, preprocessor_path = train_new_model(args, config)
                    print(f"\nModel trained and saved to {model_path}")
                except Exception as e:
                    print(f"Error training model: {str(e)}")
            
            elif choice == "3":
                # Predict prices
                try:
                    df_pred, summary = predict_prices(args, config)
                except Exception as e:
                    print(f"Error predicting prices: {str(e)}")
            
            elif choice == "4":
                # Train and predict
                try:
                    model_path, preprocessor_path = train_new_model(args, config)
                    df_pred, summary = predict_prices(args, config, model_path, preprocessor_path)
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        elif choice == "5":
            print("\nExiting application. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number from 1 to 5.")


def main():
    """Main entry point for the application."""
    # Load configuration
    config = load_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction with Phased LSTM")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", action="store_true", help="List available cryptocurrencies")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--predict", action="store_true", help="Predict prices")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Cryptocurrency symbol")
    parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"], help="Directory for saved models")
    parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"], help="Directory for prediction outputs")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode()
        return
    
    # List available cryptocurrencies if requested
    if args.list:
        list_available_cryptos()
        return
    
    # Train a new model if requested
    if args.train:
        try:
            model_path, preprocessor_path = train_new_model(args, config)
            print(f"\nModel trained and saved to {model_path}")
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    # Predict prices if requested
    if args.predict:
        try:
            df_pred, summary = predict_prices(args, config)
        except Exception as e:
            print(f"Error predicting prices: {str(e)}")


if __name__ == "__main__":
    main()