# Cryptocurrency Price Prediction with Phased LSTM

A comprehensive PyTorch-based system for predicting cryptocurrency prices using an enhanced Phased LSTM model. This application fetches data from the Binance WebSocket API, trains a sophisticated deep learning model, and generates daily open and close price predictions for any cryptocurrency available on Binance for the next 7 days.

## Features

- **Advanced Phased LSTM Architecture**: Implementation of the Phased LSTM model with attention mechanisms, residual connections, and regularization techniques
- **Real-time Data**: Fetches data from Binance WebSocket API for the most up-to-date information
- **Multi-feature Analysis**: Incorporates price, volume, and technical indicators for prediction
- **Flexible Configuration**: Easily configurable parameters for different trading pairs and strategies
- **Visualization**: Generates intuitive visualizations of predictions with historical context
- **Interactive Mode**: User-friendly command-line interface for interacting with the system
- **Easy to Maintain**: Modular code structure designed for a single engineer to maintain

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Internet connection for accessing Binance API

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-phased-lstm.git
   cd crypto-phased-lstm
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
crypto-phased-lstm/
├── config.py               # Configuration management
├── main.py                 # Main application entry point
├── predict.py              # Prediction functionality
├── train.py                # Model training functionality
├── requirements.txt        # Project dependencies
├── config.json             # Configuration file
├── data/
│   ├── binance_api.py      # Binance API interface
│   └── preprocessor.py     # Data preprocessing
├── models/
│   └── phased_lstm.py      # Phased LSTM model implementation
├── utils/
│   └── metrics.py          # Evaluation metrics
├── saved_models/           # Directory for saved models
└── predictions/            # Directory for prediction outputs
```

## Usage

### Interactive Mode

Run the application in interactive mode:

```bash
python main.py --interactive
```

This will present a menu with options to:
1. List available cryptocurrencies
2. Train a new model
3. Predict prices
4. Train and predict
5. Exit

### Command Line Options

Train a new model:

```bash
python main.py --train --symbol BTCUSDT
```

Make predictions with an existing model:

```bash
python main.py --predict --symbol BTCUSDT
```

List available cryptocurrencies:

```bash
python main.py --list
```

### Configuration

The default configuration is stored in `config.json`. You can modify this file to change parameters like:
- Training epochs, batch size, learning rate
- Model architecture parameters
- Data preprocessing settings
- Output paths

You can also optimize the configuration for specific cryptocurrencies:

```bash
python config.py --optimize BTCUSDT --target accuracy
```

## Model Architecture

The architecture employs an enhanced Phased LSTM model with several advanced features:

1. **Phased LSTM Cells**: Extends standard LSTM with time gates to handle time series data more effectively
2. **Temporal Attention**: Focuses on the most important timesteps in the sequence
3. **Residual Connections**: Allows for better gradient flow during training
4. **Layer Normalization**: Stabilizes training and improves convergence
5. **Regularization**: Multiple regularization techniques to prevent overfitting

The model takes in multiple features including:
- Price data (Open, High, Low, Close)
- Volume data
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)

## Performance Optimization

The model includes several optimizations to prevent underfitting and overfitting:

1. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving
2. **Learning Rate Scheduling**: Cosine annealing with warm restarts for better convergence
3. **Gradient Clipping**: Prevents exploding gradients
4. **L2 Regularization**: Penalizes large weights to prevent overfitting
5. **Dropout**: Applied at multiple layers with different rates
6. **Attention Mechanisms**: Helps the model focus on relevant parts of the input sequence

## Evaluation Metrics

The system evaluates predictions using multiple metrics:

1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
2. **Root Mean Squared Error (RMSE)**: Square root of the average squared differences
3. **Mean Absolute Percentage Error (MAPE)**: Percentage difference between predicted and actual prices
4. **Directional Accuracy**: Percentage of correct predictions of price movement direction

## Example Output

When making predictions, the system generates:

1. A DataFrame with predicted open and close prices for the next 7 days
2. A visualization showing historical prices and predictions
3. A JSON summary with prediction details and trend analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Phased LSTM paper: "Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences" by Daniel Neil, Michael Pfeiffer, Shih-Chii Liu
- PyTorch team for the excellent deep learning framework
- Binance for providing the WebSocket API