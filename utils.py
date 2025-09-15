import pandas as pd
import os

def load_dataset(disease_name):
    """Load dataset based on disease name"""
    file_path = f"data/{disease_name}.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    data = pd.read_csv(file_path)
    
    # Determine target column (assuming standard naming)
    target_col = 'Outcome' if 'Outcome' in data.columns else 'target'
    
    return data, target_col

def balance_dataset(original_data, synthetic_samples, target_col):
    """Combine original data with synthetic samples to create balanced dataset"""
    # Ensure synthetic samples have same columns
    synthetic_samples[target_col] = 1  # Mark as positive class
    
    # Combine datasets
    balanced_data = pd.concat([
        original_data[original_data[target_col] == 0],  # Majority class
        synthetic_samples
    ], ignore_index=True)
    
    # Shuffle the dataset
    return balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)