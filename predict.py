import os
import torch
import numpy as np
import pandas as pd
import argparse
import pickle
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.phased_lstm import CryptoPhasedLSTM

class CryptoPredictor:
    """
    Class for making cryptocurrency price predictions using trained models.
    """
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model_path: Path to the saved model checkpoint
            preprocessor_path: Path to the saved preprocessor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint['config']
        
        # Initialize model
        self.model = CryptoPhasedLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Binance API
        self.api = BinanceAPI()
    
    def predict(self, symbol, days=30):
        """
        Make predictions for the given cryptocurrency.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            days: Number of days of historical data to use
        
        Returns:
            DataFrame with predictions
        """
        # Fetch recent data
        df = self.api.get_historical_klines(
            symbol=symbol,
            interval="1d",
            limit=days + self.preprocessor.sequence_length
        )
        
        if len(df) < self.preprocessor.sequence_length:
            raise ValueError(f"Not enough historical data. Got {len(df)} days, need at least {self.preprocessor.sequence_length}.")
            
        # Process the data
        df = self.preprocessor.process_raw_data(df)
        
        # Ensure we have enough data after processing
        if len(df) < self.preprocessor.sequence_length:
            raise ValueError(f"Not enough data after preprocessing. Got {len(df)} valid data points, need {self.preprocessor.sequence_length}.")
            
        # Ensure all features are available (no NaN values in important columns)
        required_features = ["Open", "High", "Low", "Close", "Volume", "MA7", "MA14", "MA30"]
        missing_features = [col for col in required_features if col in df.columns and df[col].isna().any()]
        if missing_features:
            raise ValueError(f"Missing values in features: {missing_features}. Need more historical data.")
        
        # Prepare data for prediction
        X, T = self.preprocessor.prepare_single_prediction(df)
        
        # Verify tensor shapes
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Invalid tensor shape: X shape is {X.shape}. Check data preprocessing.")
        
        # Move tensors to device
        X, T = X.to(self.device), T.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(X, T)
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        
        # Scale back to original values
        original_predictions = self.preprocessor.inverse_transform_predictions(predictions_np)
        
        # Reshape to [open, close] format
        original_predictions = original_predictions.reshape(-1, 7, 2)
        
        # Create a DataFrame with predictions
        last_date = df["Open time"].iloc[-1]
        dates = [last_date + timedelta(days=i+1) for i in range(7)]
        
        result = []
        for i, date in enumerate(dates):
            result.append({
                "Date": date,
                "Predicted Open": original_predictions[0, i, 0],
                "Predicted Close": original_predictions[0, i, 1]
            })
        
        return pd.DataFrame(result)
    
    def create_visualization(self, df_pred, symbol, output_dir="predictions"):
        """
        Create visualization of the predictions.
        
        Args:
            df_pred: DataFrame with predictions
            symbol: Trading pair symbol
            output_dir: Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch recent historical data for context (last 30 days)
        df_hist = self.api.get_historical_klines(
            symbol=symbol,
            interval="1d",
            limit=30
        )
        
        # Select relevant columns and convert to DataFrame format
        df_hist = df_hist[["Open time", "Open", "High", "Low", "Close"]].copy()
        df_hist.loc[:, "Open"] = pd.to_numeric(df_hist["Open"])
        df_hist.loc[:, "High"] = pd.to_numeric(df_hist["High"])
        df_hist.loc[:, "Low"] = pd.to_numeric(df_hist["Low"])
        df_hist.loc[:, "Close"] = pd.to_numeric(df_hist["Close"])
        df_hist.columns = ["Date", "Open", "High", "Low", "Close"]
        
        # Calculate percentage change from last close to predicted values
        last_close = df_hist["Close"].iloc[-1]
        df_pred.loc[:, "Change_From_Last"] = ((df_pred["Predicted Close"] - last_close) / last_close * 100)
        
        # Set seaborn style
        sns.set(style="whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        
        # Plot historical prices as candlesticks
        for i in range(len(df_hist)):
            # Determine if it's a bullish or bearish candle
            if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i]:
                color = 'green'
                body_bottom = df_hist['Open'].iloc[i]
                body_top = df_hist['Close'].iloc[i]
            else:
                color = 'red'
                body_bottom = df_hist['Close'].iloc[i]
                body_top = df_hist['Open'].iloc[i]
            
            # Plot the candle body
            date = df_hist['Date'].iloc[i]
            rect = plt.Rectangle((i-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                 color=color, alpha=0.7, zorder=2)
            ax1.add_patch(rect)
            
            # Plot the high/low wicks
            ax1.plot([i, i], [df_hist['Low'].iloc[i], df_hist['High'].iloc[i]], 
                     color='black', linewidth=1, zorder=1)
        
        # Calculate MA for historical data
        df_hist.loc[:, 'MA5'] = df_hist['Close'].rolling(window=5).mean()
        df_hist.loc[:, 'MA10'] = df_hist['Close'].rolling(window=10).mean()
        
        # Plot moving averages
        x_hist = range(len(df_hist))
        if len(df_hist) >= 5:
            ax1.plot(x_hist, df_hist['MA5'].values, color='blue', linewidth=1.5, label='5-day MA')
        if len(df_hist) >= 10:
            ax1.plot(x_hist, df_hist['MA10'].values, color='orange', linewidth=1.5, label='10-day MA')
        
        # Add prediction zone shading
        hist_len = len(df_hist)
        ax1.axvspan(hist_len-1, hist_len+len(df_pred)-1, color='lightgray', alpha=0.3, label='Prediction Zone')
        
        # Draw vertical line to separate historical data and predictions
        last_hist_date = df_hist["Date"].iloc[-1]
        ax1.axvline(x=hist_len-1, color="black", linestyle="--", linewidth=1.5)
        
        # Plot predicted open/close as candlesticks
        for i in range(len(df_pred)):
            idx = hist_len + i
            # Determine if it's a bullish or bearish candle
            if df_pred['Predicted Close'].iloc[i] >= df_pred['Predicted Open'].iloc[i]:
                color = 'green'
                body_bottom = df_pred['Predicted Open'].iloc[i]
                body_top = df_pred['Predicted Close'].iloc[i]
            else:
                color = 'red'
                body_bottom = df_pred['Predicted Close'].iloc[i]
                body_top = df_pred['Predicted Open'].iloc[i]
            
            # Plot the candle body with hatch pattern for predictions
            rect = plt.Rectangle((idx-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                color=color, alpha=0.5, hatch='///', zorder=2)
            ax1.add_patch(rect)
            
            # Connect predicted candles with lines
            if i > 0:
                prev_idx = hist_len + i - 1
                ax1.plot([prev_idx, idx], 
                         [df_pred['Predicted Close'].iloc[i-1], df_pred['Predicted Open'].iloc[i]],
                         color='blue', linestyle=':', linewidth=1)
        
        # Predicted price range
        pred_min = df_pred[['Predicted Open', 'Predicted Close']].min().min()
        pred_max = df_pred[['Predicted Open', 'Predicted Close']].max().max()
        hist_min = df_hist[['Open', 'Close', 'Low']].min().min()
        hist_max = df_hist[['Open', 'Close', 'High']].max().max()
        
        # Calculate overall price range and add padding
        y_min = min(hist_min, pred_min)
        y_max = max(hist_max, pred_max)
        y_range = y_max - y_min
        y_min -= y_range * 0.05
        y_max += y_range * 0.05
        
        # Set y-axis limits with padding
        ax1.set_ylim(y_min, y_max)
        
        # Add price labels on the right for predicted values
        for i, row in df_pred.iterrows():
            idx = hist_len + i
            price = row['Predicted Close']
            color = 'green' if row['Predicted Close'] >= row['Predicted Open'] else 'red'
            change = row['Change_From_Last']
            change_sign = '+' if change >= 0 else ''
            ax1.annotate(f"{price:.2f} ({change_sign}{change:.2f}%)", 
                        xy=(idx, price),
                        xytext=(idx + 0.1, price),
                        fontsize=9,
                        color=color)
        
        # Format x-axis with dates
        all_dates = list(df_hist['Date']) + list(df_pred['Date'])
        ax1.set_xticks(range(len(all_dates)))
        
        # Only show a selection of dates to avoid crowding
        show_dates = []
        date_labels = []
        date_step = max(1, len(all_dates) // 10)  # Show at most 10 dates
        for i in range(0, len(all_dates), date_step):
            show_dates.append(i)
            date_labels.append(all_dates[i].strftime('%Y-%m-%d'))
            
        # Always include the last historical date and last prediction date
        if hist_len - 1 not in show_dates:
            show_dates.append(hist_len - 1)
            date_labels.append(all_dates[hist_len - 1].strftime('%Y-%m-%d'))
        if len(all_dates) - 1 not in show_dates:
            show_dates.append(len(all_dates) - 1)
            date_labels.append(all_dates[-1].strftime('%Y-%m-%d'))
            
        # Sort and apply labels
        show_dates, date_labels = zip(*sorted(zip(show_dates, date_labels)))
        ax1.set_xticks(show_dates)
        ax1.set_xticklabels(date_labels, rotation=45, ha='right')
        
        # Volume subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot historical volume
        for i in range(len(df_hist)):
            color = 'green' if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i] else 'red'
            ax2.bar(i, df_hist['Volume'].iloc[i] if 'Volume' in df_hist.columns else 0, 
                   color=color, alpha=0.5, width=0.8)
        
        # Add volume title
        ax2.set_ylabel('Volume')
        ax2.tick_params(axis='x', labelbottom=False)
        
        # Daily change subplot
        ax3 = fig.add_subplot(gs[2])
        
        # Calculate historical daily changes
        if len(df_hist) > 1:
            df_hist.loc[:, 'Daily_Change'] = df_hist['Close'].pct_change() * 100
            
            # Plot historical daily changes
            for i in range(1, len(df_hist)):
                change = df_hist['Daily_Change'].iloc[i]
                color = 'green' if change >= 0 else 'red'
                ax3.bar(i, change, color=color, alpha=0.7, width=0.8)
        
        # Plot predicted daily changes
        for i in range(len(df_pred)):
            idx = hist_len + i
            if i == 0:
                # First day compared to last historical close
                change = ((df_pred['Predicted Close'].iloc[i] - df_hist['Close'].iloc[-1]) / 
                          df_hist['Close'].iloc[-1] * 100)
            else:
                # Other days compared to previous predicted close
                change = ((df_pred['Predicted Close'].iloc[i] - df_pred['Predicted Close'].iloc[i-1]) / 
                          df_pred['Predicted Close'].iloc[i-1] * 100)
            
            color = 'green' if change >= 0 else 'red'
            ax3.bar(idx, change, color=color, alpha=0.5, width=0.8, hatch='///')
            
            # Add percentage labels
            ax3.annotate(f"{change:.2f}%", 
                        xy=(idx, change),
                        xytext=(idx, change + (0.5 if change >= 0 else -0.5)),
                        fontsize=8,
                        ha='center',
                        va='bottom' if change >= 0 else 'top',
                        color=color)
        
        # Set daily change y-axis
        max_change = max(3, 
                         df_hist['Daily_Change'].abs().max() if 'Daily_Change' in df_hist.columns else 0,
                         df_pred['Change_From_Last'].abs().max() if 'Change_From_Last' in df_pred.columns else 0)
        ax3.set_ylim(-max_change * 1.2, max_change * 1.2)
        ax3.set_ylabel('Daily Change %')
        
        # Set x-axis limits
        ax1.set_xlim(-0.5, len(all_dates) - 0.5)
        
        # Add labels and title
        ax1.set_ylabel('Price')
        ax3.set_xlabel('Date')
        ax1.set_title(f"{symbol} Price Prediction for Next 7 Days", fontsize=16)
        ax1.legend()
        
        # Add annotations
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall_change = df_pred['Change_From_Last'].iloc[-1]
        change_text = f"Predicted 7-day change: {'+' if overall_change >= 0 else ''}{overall_change:.2f}%"
        fig.text(0.02, 0.02, f"Prediction made on: {prediction_date}\n{change_text}", 
                fontsize=10, color="gray")
        
        # Add prediction summary
        avg_pred = df_pred["Predicted Close"].mean()
        min_pred = df_pred["Predicted Close"].min()
        max_pred = df_pred["Predicted Close"].max()
        trend = "Bullish" if df_pred["Predicted Close"].iloc[-1] > df_hist["Close"].iloc[-1] else "Bearish"
        
        summary_text = (
            f"Prediction Summary:\n"
            f"Current price: {df_hist['Close'].iloc[-1]:.2f}\n"
            f"Average predicted: {avg_pred:.2f}\n"
            f"Range: {min_pred:.2f} - {max_pred:.2f}\n"
            f"7-day outlook: {trend}"
        )
        
        # Add text box for summary
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"{symbol}_prediction_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Visualization saved to {plot_path}")
        
        return plot_path
    
    def get_prediction_summary(self, df_pred, symbol):
        """
        Generate a summary of the predictions.
        
        Args:
            df_pred: DataFrame with predictions
            symbol: Trading pair symbol
        
        Returns:
            Dictionary with prediction summary
        """
        # Calculate price changes
        first_open = df_pred["Predicted Open"].iloc[0]
        last_close = df_pred["Predicted Close"].iloc[-1]
        price_change = last_close - first_open
        price_change_pct = (price_change / first_open) * 100
        
        # Calculate daily changes
        daily_changes = []
        for i in range(len(df_pred)):
            open_price = df_pred["Predicted Open"].iloc[i]
            close_price = df_pred["Predicted Close"].iloc[i]
            daily_change = close_price - open_price
            daily_change_pct = (daily_change / open_price) * 100
            daily_changes.append({
                "date": df_pred["Date"].iloc[i].strftime("%Y-%m-%d"),
                "open": open_price,
                "close": close_price,
                "change": daily_change,
                "change_pct": daily_change_pct
            })
        
        # Check if trend is bullish, bearish, or sideways
        if price_change_pct > 3:
            trend = "Bullish"
        elif price_change_pct < -3:
            trend = "Bearish"
        else:
            trend = "Sideways"
        
        # Create summary
        summary = {
            "symbol": symbol,
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_period": {
                "start": df_pred["Date"].iloc[0].strftime("%Y-%m-%d"),
                "end": df_pred["Date"].iloc[-1].strftime("%Y-%m-%d")
            },
            "overall": {
                "start_price": first_open,
                "end_price": last_close,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "trend": trend
            },
            "daily_predictions": daily_changes
        }
        
        return summary


def predict_crypto_prices(model_path, preprocessor_path, symbol, visualize=True, output_dir="predictions"):
    """
    Predict cryptocurrency prices using a trained model.
    
    Args:
        model_path: Path to the saved model checkpoint
        preprocessor_path: Path to the saved preprocessor
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        visualize: Whether to create visualization
        output_dir: Directory to save the results
    """
    # Create predictor
    predictor = CryptoPredictor(model_path, preprocessor_path)
    
    # Make predictions
    df_pred = predictor.predict(symbol)
    
    # Print predictions
    print("\nPredictions for", symbol)
    print(df_pred.to_string(index=False))
    
    # Create visualization if requested
    if visualize:
        plot_path = predictor.create_visualization(df_pred, symbol, output_dir)
    
    # Generate summary
    summary = predictor.get_prediction_summary(df_pred, symbol)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Symbol: {summary['symbol']}")
    print(f"Prediction Period: {summary['prediction_period']['start']} to {summary['prediction_period']['end']}")
    print(f"Overall Trend: {summary['overall']['trend']}")
    print(f"Price Change: {summary['overall']['price_change']:.2f} ({summary['overall']['price_change_pct']:.2f}%)")
    
    # Save summary to JSON
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"{symbol}_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to {summary_path}")
    
    return df_pred, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cryptocurrency prices')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--preprocessor', type=str, required=True, help='Path to the preprocessor')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    
    args = parser.parse_args()
    
    predict_crypto_prices(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        symbol=args.symbol,
        visualize=args.visualize,
        output_dir=args.output_dir
    )