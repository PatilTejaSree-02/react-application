import pandas as pd
from config import DISEASE_CONFIGS

def load_data(disease_name):
    """Load and preprocess dataset"""
    if disease_name not in DISEASE_CONFIGS:
        raise ValueError(f"No configuration found for {disease_name}")
    
    config = DISEASE_CONFIGS[disease_name]
    data = pd.read_csv(config["url"], names=config["columns"])
    
    # Convert to numeric and handle missing values
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    return data, config["target"], config["positive_class"]

def check_class_distribution(data, target_col):
    """Display class distribution"""
    print("\nClass Distribution:")
    print(data[target_col].value_counts().to_frame('Count'))