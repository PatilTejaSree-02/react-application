import os
import time
import uuid  # For generating unique IDs
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from config import DISEASE_CONFIGS
from data_loader import load_data, check_class_distribution
from synthesizer import generate_synthetic_samples
from evaluator import evaluate_models  # Assuming this returns a dict with reports
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import confusion_matrix
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('metadata', exist_ok=True)

# Define the list of diseases
DISEASES_LIST = [
    {"name": "diabetes", "requires_upload": True, "target_column": "Outcome", "positive_class": 1},
    {"name": "heart_disease", "requires_upload": True, "target_column": "target", "positive_class": 1},
    # Add more diseases as needed
]

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    return jsonify(DISEASES_LIST)

def train_and_evaluate_model(data, target_col):
    """Trains a simple model and returns the confusion matrix."""
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(solver='liblinear', random_state=42)  # Example model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm.tolist()  # Convert to list for JSON serialization

def run_pipeline_on_upload(file_path, disease_config):
    """Runs the pipeline on a user-uploaded file."""
    print(f"\n{'='*50}\n🔵 Processing user-uploaded data for {disease_config['name']}\n{'='*50}")

    try:
        # --- DELETE METADATA.JSON AT THE BEGINNING ---
        metadata_file_path = 'metadata/metadata.json'
        if os.path.exists(metadata_file_path):
            print(f"Attempting to delete: {metadata_file_path}")
            os.remove(metadata_file_path)
            print(f"✓ Deleted existing metadata file: {metadata_file_path}")
        else:
            print(f"Metadata file not found at: {metadata_file_path}")
        # ---------------------------------------------

        # 1. Data Loading
        print("\n[1/5] Loading uploaded data...")
        start_time = time.time()
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error loading uploaded file: {e}")

        target_col = disease_config["target_column"]
        positive_class = disease_config["positive_class"]
        check_class_distribution(data, target_col)
        print(f"✓ Loaded {len(data)} samples in {time.time()-start_time:.1f}s")

        # 2. Confusion Matrix on Original Data
        print("\n[2/5] Generating confusion matrix on original data...")
        cm_original = train_and_evaluate_model(data.copy(), target_col)

        # 3. Synthetic Generation
        print("\n[3/5] Generating synthetic samples...")
        minority_count = sum(data[target_col] == positive_class)
        samples_to_generate = sum(data[target_col] == 0) - minority_count # Generate enough to match majority

        if samples_to_generate > 0:
            try:
                synth_start = time.time()
                synthetic_samples = generate_synthetic_samples(
                    data.copy(),
                    target_col,
                    positive_class,
                    samples_to_generate,
                    # metadata_path='metadata/metadata.json' # Removed unique path
                )
                print(f"✓ Generated {len(synthetic_samples)} synthetic samples in {time.time()-synth_start:.1f}s")
            except Exception as e:
                print(f"⚠️ Primary generation failed: {str(e)}")
                print("Attempting fallback generation (generating up to needed)...")
                synthetic_samples = generate_synthetic_samples(
                    data.copy(),
                    target_col,
                    positive_class,
                    samples_to_generate=samples_to_generate,
                    # metadata_path='metadata/metadata.json' # Removed unique path
                )
                print(f"✓ Generated {len(synthetic_samples)} (fallback) samples")
        else:
            synthetic_samples = pd.DataFrame()
            print("Majority class is not larger than minority, skipping synthetic generation for balancing.")

        # 4. Data Balancing (Ensuring equal counts)
        print("\n[4/5] Creating balanced dataset with equal class counts...")
        majority_data = data[data[target_col] != positive_class]
        minority_data = data[data[target_col] == positive_class]

        num_majority = len(majority_data)
        num_minority = len(minority_data)
        needed_synthetic = num_majority - num_minority

        synthetic_to_add = synthetic_samples.head(max(0, needed_synthetic))

        balanced_data = pd.concat([majority_data, minority_data, synthetic_to_add]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✓ Final class distribution:")
        check_class_distribution(balanced_data, target_col)
        print(f"  - Class 0 count: {sum(balanced_data[target_col] == 0)}")
        print(f"  - Class 1 count: {sum(balanced_data[target_col] == 1)}")

        # Save balanced results (still good for download)
        unique_id = uuid.uuid4().hex # Keep unique ID for results filename
        balanced_file_path = f"results/balanced_uploaded_{disease_config['name']}_{unique_id}.csv" # Include unique ID
        balanced_data.to_csv(balanced_file_path, index=False)

        # 5. Confusion Matrix on Balanced Data and Evaluation
        print("\n[5/5] Generating confusion matrix on balanced data and evaluating...")
        cm_balanced = train_and_evaluate_model(balanced_data.copy(), target_col)
        eval_start = time.time()
        evaluation_results = evaluate_models(data.copy(), balanced_data.copy(), target_col, disease_config['name'])
        print(f"✓ Evaluation completed in {time.time()-eval_start:.1f}s")

        # Pipeline summary
        total_time = time.time() - start_time
        print(f"\n✅ Pipeline completed in {total_time:.1f} seconds")
        print(f"Results saved to: {balanced_file_path}")
        return balanced_file_path, {'evaluation': evaluation_results, 'confusion_matrix_original': cm_original, 'confusion_matrix_balanced': cm_balanced}

    except Exception as e:
        print(f"\n🔴 Critical pipeline failure: {str(e)}")
        return None, {'error': str(e)}

@app.route('/api/upload/<disease>', methods=['POST'])
def upload_file(disease):
    if disease not in [d['name'] for d in DISEASES_LIST]:
        return jsonify({'error': 'Invalid disease name'}), 400

    disease_config = next((d for d in DISEASES_LIST if d['name'] == disease), None)
    if not disease_config:
        return jsonify({'error': f'Configuration not found for {disease}'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        results_path, results = run_pipeline_on_upload(file_path, disease_config)
        os.remove(file_path)  # Clean up uploaded file
        if results_path:
            return jsonify({'results_path': results_path, 'evaluation': results['evaluation'],
                            'confusion_matrix_original': results['confusion_matrix_original'],
                            'confusion_matrix_balanced': results['confusion_matrix_balanced']})
        else:
            return jsonify({'error': 'Pipeline failed', 'details': results.get('error')}), 500

@app.route('/results/preview/<filename>', methods=['GET'])
def preview_result(filename):
    """Endpoint to preview the generated balanced CSV data."""
    file_path = os.path.join('results', filename)
    try:
        df = pd.read_csv(file_path)
        return jsonify(df.head().to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

@app.route('/results/download/<filename>', methods=['GET'])
def download_result(filename):
    """Endpoint to download the generated balanced CSV data."""
    file_path = os.path.join('results', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)