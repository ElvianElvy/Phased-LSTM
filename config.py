import os
import json
import argparse
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "model_dir": "saved_models",
        "output_dir": "predictions",
        "log_dir": "logs"
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_size": 256,  # Increased from default
        "num_layers": 3,     # Increased from default
        "sequence_length": 30,
        "train_days": 365,
        "validation_split": 0.2
    },
    "prediction": {
        "create_visualization": True,
        "prediction_days": 7
    },
    "model": {
        "dropout": 0.3,      # Increased from default
        "l2_reg": 1e-5,
        "use_attention": True
    }
}

CONFIG_FILE = "config.json"


def create_default_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)) if os.path.dirname(config_path) else '.', exist_ok=True)
    
    # Write the default config to the file
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    
    print(f"Created default configuration file: {config_path}")
    return DEFAULT_CONFIG


def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration from file or create default if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    # Check if file exists and has content
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        return create_default_config(config_path)
    
    # Try to load the config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError:
        # If there's an error decoding JSON, create a default config
        print(f"Error parsing {config_path}. Creating default configuration.")
        return create_default_config(config_path)


def update_config(updates: Dict[str, Any], config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of updates
        config_path: Path to the configuration file
    
    Returns:
        Updated configuration dictionary
    """
    # Load current config
    config = load_config(config_path)
    
    # Recursive function to update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict(d[k], v)
            else:
                d[k] = v
    
    # Update config
    update_dict(config, updates)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config


def optimize_config(symbol: str, optimize_for: str = "accuracy") -> Dict[str, Any]:
    """
    Optimize configuration for a specific cryptocurrency.
    
    Args:
        symbol: Trading pair symbol
        optimize_for: Optimization target ("accuracy" or "speed")
    
    Returns:
        Optimized configuration dictionary
    """
    # Load default config
    config = load_config()
    
    # Optimize based on target
    if optimize_for == "accuracy":
        config["training"]["epochs"] = 150
        config["training"]["batch_size"] = 32
        config["training"]["learning_rate"] = 0.0005
        config["training"]["hidden_size"] = 384
        config["training"]["num_layers"] = 3
        config["training"]["sequence_length"] = 45
        config["training"]["train_days"] = 730  # 2 years
        config["model"]["dropout"] = 0.3
        config["model"]["l2_reg"] = 1e-5
    
    elif optimize_for == "speed":
        config["training"]["epochs"] = 50
        config["training"]["batch_size"] = 64
        config["training"]["learning_rate"] = 0.001
        config["training"]["hidden_size"] = 128
        config["training"]["num_layers"] = 2
        config["training"]["sequence_length"] = 20
        config["training"]["train_days"] = 365  # 1 year
        config["model"]["dropout"] = 0.2
        config["model"]["l2_reg"] = 1e-6
    
    # Some symbol-specific optimizations (just examples)
    if symbol == "BTCUSDT" or symbol == "ETHUSDT":
        # Major cryptocurrencies might benefit from more data and complexity
        config["training"]["train_days"] = 1095  # 3 years
        config["training"]["hidden_size"] = 512 if optimize_for == "accuracy" else 256
    
    elif symbol.endswith("BTC"):
        # Pairs traded against BTC might have different patterns
        config["training"]["sequence_length"] = 40 if optimize_for == "accuracy" else 25
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction Configuration")
    parser.add_argument("--create", action="store_true", help="Create default configuration file")
    parser.add_argument("--view", action="store_true", help="View current configuration")
    parser.add_argument("--optimize", type=str, help="Optimize configuration for symbol")
    parser.add_argument("--target", type=str, choices=["accuracy", "speed"], default="accuracy",
                        help="Optimization target")
    
    args = parser.parse_args()
    
    if args.create:
        create_default_config()
        print("Default configuration created.")
    
    if args.view:
        config = load_config()
        print(json.dumps(config, indent=4))
    
    if args.optimize:
        config = optimize_config(args.optimize, args.target)
        print(f"Optimized configuration for {args.optimize} (target: {args.target}):")
        print(json.dumps(config, indent=4))