import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import DISEASE_CONFIGS

def evaluate_models(original_data, augmented_data, target_col, disease_name):
    """Compare original vs augmented model performance and return metrics."""
    try:
        # Prepare data
        X_orig = original_data.drop(target_col, axis=1)
        y_orig = original_data[target_col]

        # Class counts for original data
        original_counts = y_orig.value_counts().to_dict()
        original_count_0 = original_counts.get(0, 0)
        original_count_1 = original_counts.get(1, 0)

        # Split original data (same test set for both)
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X_orig, y_orig,
            test_size=DISEASE_CONFIGS[disease_name]["test_size"],
            random_state=42,
            stratify=y_orig
        )

        # Model 1: Original data only
        orig_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        orig_model.fit(X_train_orig, y_train_orig)
        orig_accuracy = accuracy_score(y_test, orig_model.predict(X_test))
        orig_report = classification_report(y_test, orig_model.predict(X_test), zero_division=0, output_dict=True)

        # Model 2: Augmented data
        X_train_aug = pd.concat([
            X_train_orig,
            augmented_data[augmented_data[target_col] == 1].drop(target_col, axis=1)
        ])
        y_train_aug = pd.concat([
            y_train_orig,
            augmented_data[augmented_data[target_col] == 1][target_col]
        ])

        # Class counts for augmented data
        augmented_counts = augmented_data[target_col].value_counts().to_dict()
        augmented_count_0 = augmented_counts.get(0, 0)
        augmented_count_1 = augmented_counts.get(1, 0)

        aug_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        aug_model.fit(X_train_aug, y_train_aug)
        aug_accuracy = accuracy_score(y_test, aug_model.predict(X_test))
        aug_report = classification_report(y_test, aug_model.predict(X_test), zero_division=0, output_dict=True)

        # Plot feature importance and save to a file
        feature_importance_path = plot_feature_importance(orig_model, aug_model, X_train_orig.columns)

        return {
            'original_accuracy': orig_accuracy,
            'augmented_accuracy': aug_accuracy,
            'original_report': orig_report,
            'augmented_report': aug_report,
            'feature_importance_path': feature_importance_path,
            'original_count': len(original_data),
            'augmented_count': len(augmented_data),
            'original_count_0': original_count_0,
            'original_count_1': original_count_1,
            'augmented_count_0': augmented_count_0,
            'augmented_count_1': augmented_count_1,
        }

    except Exception as e:
        print(f"\n🔴 Evaluation failed: {str(e)}")
        raise

def plot_feature_importance(orig_model, aug_model, feature_names):
    """Compare feature importance between models and save the plot."""
    plt.figure(figsize=(15, 6))

    # Original model
    plt.subplot(1, 2, 1)
    orig_importances = orig_model.feature_importances_
    indices = np.argsort(orig_importances)[::-1]
    plt.title("Original Model Feature Importance")
    plt.bar(range(len(indices)), orig_importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)

    # Augmented model
    plt.subplot(1, 2, 2)
    aug_importances = aug_model.feature_importances_
    indices = np.argsort(aug_importances)[::-1]
    plt.title("Augmented Model Feature Importance")
    plt.bar(range(len(indices)), aug_importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive, for saving files)
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import DISEASE_CONFIGS

def evaluate_models(original_data, augmented_data, target_col, disease_name):
    """Compare original vs augmented model performance and return metrics."""
    try:
        # Prepare data
        X_orig = original_data.drop(target_col, axis=1)
        y_orig = original_data[target_col]

        # Class counts for original data
        original_counts = y_orig.value_counts().to_dict()
        original_count_0 = original_counts.get(0, 0)
        original_count_1 = original_counts.get(1, 0)

        # Split original data (same test set for both)
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X_orig, y_orig,
            test_size=DISEASE_CONFIGS[disease_name]["test_size"],
            random_state=42,
            stratify=y_orig
        )

        # Model 1: Original data only
        orig_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        orig_model.fit(X_train_orig, y_train_orig)
        orig_accuracy = accuracy_score(y_test, orig_model.predict(X_test))
        orig_report = classification_report(y_test, orig_model.predict(X_test), zero_division=0, output_dict=True)

        # Model 2: Augmented data
        X_train_aug = pd.concat([
            X_train_orig,
            augmented_data[augmented_data[target_col] == 1].drop(target_col, axis=1)
        ])
        y_train_aug = pd.concat([
            y_train_orig,
            augmented_data[augmented_data[target_col] == 1][target_col]
        ])

        # Class counts for augmented data
        augmented_counts = augmented_data[target_col].value_counts().to_dict()
        augmented_count_0 = augmented_counts.get(0, 0)
        augmented_count_1 = augmented_counts.get(1, 0)

        aug_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        aug_model.fit(X_train_aug, y_train_aug)
        aug_accuracy = accuracy_score(y_test, aug_model.predict(X_test))
        aug_report = classification_report(y_test, aug_model.predict(X_test), zero_division=0, output_dict=True)

        # Plot feature importance and save to a file
        feature_importance_path = plot_feature_importance(orig_model, aug_model, X_train_orig.columns)

        return {
            'original_accuracy': orig_accuracy,
            'augmented_accuracy': aug_accuracy,
            'original_report': orig_report,
            'augmented_report': aug_report,
            'feature_importance_path': feature_importance_path,
            'original_count': len(original_data),
            'augmented_count': len(augmented_data),
            'original_count_0': original_count_0,
            'original_count_1': original_count_1,
            'augmented_count_0': augmented_count_0,
            'augmented_count_1': augmented_count_1,
        }

    except Exception as e:
        print(f"\n🔴 Evaluation failed: {str(e)}")
        raise

def plot_feature_importance(orig_model, aug_model, feature_names):
    """Compare feature importance between models and save the plot."""
    plt.figure(figsize=(15, 6))

    # Original model
    plt.subplot(1, 2, 1)
    orig_importances = orig_model.feature_importances_
    indices = np.argsort(orig_importances)[::-1]
    plt.title("Original Model Feature Importance")
    plt.bar(range(len(indices)), orig_importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)

    # Augmented model
    plt.subplot(1, 2, 2)
    aug_importances = aug_model.feature_importances_
    indices = np.argsort(aug_importances)[::-1]
    plt.title("Augmented Model Feature Importance")
    plt.bar(range(len(indices)), aug_importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)

    plt.tight_layout()
    filename = 'results/feature_importance.png'
    plt.savefig(filename)
    plt.close()
    return filename
    plt.tight_layout()
    filename = 'results/feature_importance.png'
    plt.savefig(filename)
    plt.close()
    return filename