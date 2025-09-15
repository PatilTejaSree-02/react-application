import pandas as pd
from data_loader import load_data, check_class_distribution
from synthesizer import generate_synthetic_samples
from evaluator import evaluate_models
import os
from config import DISEASE_CONFIGS
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress SDV warnings

def run_pipeline(disease_name):
    """Main pipeline with enhanced error handling and progress tracking"""
    print(f"\n{'='*50}\n🔵 Processing {disease_name}\n{'='*50}")
    
    try:
        # 1. Data Loading
        print("\n[1/4] Loading data...")
        start_time = time.time()
        data, target_col, positive_class = load_data(disease_name)
        check_class_distribution(data, target_col)
        print(f"✓ Loaded {len(data)} samples in {time.time()-start_time:.1f}s")

        # 2. Synthetic Generation
        print("\n[2/4] Generating synthetic samples...")
        minority_count = sum(data[target_col] == positive_class)
        samples_to_generate = minority_count * 2  # Target 2x oversampling
        
        try:
            synth_start = time.time()
            synthetic_samples = generate_synthetic_samples(
                data, 
                target_col, 
                positive_class,
                samples_to_generate
            )
            print(f"✓ Generated {len(synthetic_samples)} synthetic samples in {time.time()-synth_start:.1f}s")
        except Exception as e:
            print(f"⚠️ Primary generation failed: {str(e)}")
            print("Attempting fallback generation...")
            synthetic_samples = generate_synthetic_samples(
                data,
                target_col,
                positive_class,
                samples_to_generate=minority_count  # Fallback to 1x
            )
            print(f"✓ Generated {len(synthetic_samples)} (fallback) samples")

        # 3. Data Balancing
        print("\n[3/4] Creating balanced dataset...")
        balanced_data = pd.concat([
            data[data[target_col] != positive_class],  # Original majority
            data[data[target_col] == positive_class],  # Original minority
            synthetic_samples  # Synthetic minority
        ]).sample(frac=1, random_state=42)  # Shuffle
        
        # Save results
        os.makedirs("results", exist_ok=True)
        balanced_data.to_csv(f"results/balanced_{disease_name}.csv", index=False)
        print(f"✓ Final class distribution:")
        check_class_distribution(balanced_data, target_col)

        # 4. Evaluation
        print("\n[4/4] Evaluating models...")
        eval_start = time.time()
        evaluate_models(data, balanced_data, target_col, disease_name)
        print(f"✓ Evaluation completed in {time.time()-eval_start:.1f}s")

        # Pipeline summary
        total_time = time.time() - start_time
        print(f"\n✅ Pipeline completed in {total_time:.1f} seconds")
        print(f"Results saved to: results/balanced_{disease_name}.csv")

    except Exception as e:
        print(f"\n🔴 Critical pipeline failure: {str(e)}")
        print("Debugging tips:")
        print("- Check data/ folder exists with correct CSV")
        print("- Verify sufficient RAM (CTGAN needs 8GB+)")
        print("- Try reducing synthetic samples count")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Medical Data Augmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--disease', 
        choices=list(DISEASE_CONFIGS.keys()), 
        required=True,
        help="Disease to process (from config.py)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.disease)