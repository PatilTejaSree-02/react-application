import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from config import DISEASE_CONFIGS

def preprocess_for_ctgan(data):
    """Safer preprocessing pipeline"""
    processed = data.copy()
    
    # Convert binary features
    binary_cols = [col for col in processed.columns 
                  if processed[col].nunique() == 2]
    for col in binary_cols:
        processed[col] = processed[col].astype(int)  # CTGAN prefers int over bool
    
    # Scale continuous features
    continuous_cols = [col for col in processed.columns 
                      if col not in binary_cols and processed[col].dtype in ['int64', 'float64']]
    if continuous_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))  # More stable than PowerTransformer
        processed[continuous_cols] = scaler.fit_transform(processed[continuous_cols])
    
    return processed

def generate_synthetic_samples(data, target_col, positive_class, samples_to_generate):
    """More robust synthetic sample generation"""
    minority_data = data[data[target_col] == positive_class].drop(target_col, axis=1)
    
    # Safer preprocessing
    processed_data = preprocess_for_ctgan(minority_data)
    
    # Configure metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(processed_data)
    
    # Stable CTGAN configuration
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100,  # Reduced from 500 for faster testing
        batch_size=256,  # Must be divisible by pac
        pac=8,  # Reduced from 10 for better compatibility
        cuda=False,
        generator_dim=(128, 128),  # Simplified architecture
        discriminator_dim=(128, 128),
        verbose=True
    )
    
    # Save metadata for reproducibility (avoid warning)
    metadata.save_to_json('metadata.json')
    
    # Train with error handling
    try:
        synthesizer.fit(processed_data)
    except Exception as e:
        print(f"⚠️ CTGAN fit failed: {str(e)}")
        print("Trying with reduced batch size...")
        synthesizer.batch_size = 128
        synthesizer.fit(processed_data)
    
    # Generate samples
    synthetic_samples = synthesizer.sample(num_rows=samples_to_generate)
    synthetic_samples[target_col] = positive_class
    
    return synthetic_samples[data.columns]  # Maintain original column order