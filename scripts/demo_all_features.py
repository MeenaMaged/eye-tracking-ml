#!/usr/bin/env python3
"""
Black Box Demo Script - Eye Tracking ML Project
================================================
Run this script section by section to capture screenshots for your report.

Usage:
  python scripts/demo_all_features.py [section]

Sections:
  1 - Gaze Estimation (Regression Models)
  2 - Blink Detection (Binary Classifiers)
  3 - Eye Movement Classification (3-Class)
  4 - Feature Selection (PCA Analysis)
  5 - Evaluation Metrics Demo
  6 - Cross-Validation Demo
  all - Run all sections

Each section will pause for you to capture screenshots.
"""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


INTERACTIVE = True  # Set to False for non-interactive mode

def wait_for_screenshot(message="Press Enter to continue after taking screenshot..."):
    """Pause for screenshot capture."""
    print("\n" + "=" * 60)
    print(">>> SCREENSHOT OPPORTUNITY <<<")
    print(message)
    print("=" * 60)
    if INTERACTIVE:
        input()


def section_header(num, title):
    """Print section header."""
    print("\n")
    print("#" * 70)
    print(f"# SECTION {num}: {title}")
    print("#" * 70)
    print()


# =============================================================================
# SECTION 1: GAZE ESTIMATION
# =============================================================================
def demo_gaze_estimation():
    """Demo gaze estimation regression models."""
    section_header(1, "GAZE ESTIMATION (REGRESSION MODELS)")

    print("This section demonstrates gaze estimation using regression models.")
    print("Input: 10-dimensional eye feature vector")
    print("Output: (x, y) screen coordinates in pixels")
    print()

    # Generate synthetic calibration data
    from gaze_estimation import GazeModelComparison

    np.random.seed(42)
    n_samples = 400

    # Feature vector: [iris_x_L, iris_y_L, iris_x_R, iris_y_R, EAR_L, EAR_R, pitch, yaw, roll, iod]
    X = np.random.randn(n_samples, 10)

    # Non-linear gaze mapping (realistic)
    y_x = 960 + 300 * X[:, 0] + 250 * X[:, 2] + 50 * X[:, 0] * X[:, 2]
    y_y = 540 + 200 * X[:, 1] + 180 * X[:, 3] + 40 * X[:, 1] * X[:, 3]
    y = np.column_stack([y_x, y_y]) + np.random.randn(n_samples, 2) * 25

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print()

    # Run comparison
    comparison = GazeModelComparison(normalize=True)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_nn=True, nn_epochs=100, verbose=True)

    print("\n")
    comparison.print_results()

    wait_for_screenshot("Screenshot the regression model comparison table above")

    # Show individual model predictions
    print("\n--- Individual Model Predictions (First 5 Test Samples) ---")
    print(f"{'True X':>8} {'True Y':>8} | {'Pred X':>8} {'Pred Y':>8} | Error (px)")
    print("-" * 60)

    best_result = comparison.get_best_model()
    best_model = best_result.model
    y_pred = best_model.predict(X_test[:5])

    for i in range(5):
        error = np.sqrt((y_test[i, 0] - y_pred[i, 0])**2 + (y_test[i, 1] - y_pred[i, 1])**2)
        print(f"{y_test[i, 0]:8.1f} {y_test[i, 1]:8.1f} | {y_pred[i, 0]:8.1f} {y_pred[i, 1]:8.1f} | {error:6.1f}")

    wait_for_screenshot("Screenshot the prediction examples above")


# =============================================================================
# SECTION 2: BLINK DETECTION
# =============================================================================
def demo_blink_detection():
    """Demo blink detection binary classifiers."""
    section_header(2, "BLINK DETECTION (BINARY CLASSIFICATION)")

    print("This section demonstrates blink detection using binary classifiers.")
    print("Input: 19-dimensional feature vector (EAR + temporal features)")
    print("Output: 0 (No Blink) or 1 (Blink)")
    print()

    from classification import ClassifierComparison

    np.random.seed(42)
    n_samples = 600

    # Generate blink features (19 dimensions)
    X = np.random.randn(n_samples, 19)

    # First feature is current EAR (Eye Aspect Ratio)
    X[:n_samples//2, 0] = np.random.uniform(0.25, 0.35, n_samples//2)  # Open eyes
    X[n_samples//2:, 0] = np.random.uniform(0.05, 0.15, n_samples//2)  # Blink

    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print()

    # Run comparison
    comparison = ClassifierComparison(normalize=True, task='blink')
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=50, verbose=True)

    print("\n")
    comparison.print_results()

    wait_for_screenshot("Screenshot the classifier comparison table above")

    # Show confusion matrix for best model
    from evaluation import confusion_matrix, print_confusion_matrix

    print("\n--- Confusion Matrix (Best Model) ---")
    best_result = comparison.get_best_model()
    best_model = best_result.model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, n_classes=2)
    print_confusion_matrix(cm, class_names=['No Blink', 'Blink'])

    wait_for_screenshot("Screenshot the confusion matrix above")


# =============================================================================
# SECTION 3: EYE MOVEMENT CLASSIFICATION
# =============================================================================
def demo_movement_classification():
    """Demo 3-class eye movement classification."""
    section_header(3, "EYE MOVEMENT CLASSIFICATION (3-CLASS)")

    print("This section demonstrates eye movement classification.")
    print("Input: 14-dimensional feature vector (velocity + dispersion + EAR)")
    print("Output: 0 (Fixation), 1 (Saccade), 2 (Blink)")
    print()

    from classification import MovementClassifierComparison

    np.random.seed(42)
    n_samples = 600

    # Generate movement features (14 dimensions)
    X = np.random.randn(n_samples, 14)

    n_fix = n_samples // 2      # 50% fixation
    n_sac = n_samples // 3      # 33% saccade
    n_blink = n_samples - n_fix - n_sac  # 17% blink

    # Fixation: low velocity, normal EAR
    X[:n_fix, 0] = np.random.uniform(0, 10, n_fix)          # velocity
    X[:n_fix, 8] = np.random.uniform(0.25, 0.35, n_fix)     # EAR

    # Saccade: high velocity, normal EAR
    X[n_fix:n_fix+n_sac, 0] = np.random.uniform(50, 200, n_sac)
    X[n_fix:n_fix+n_sac, 8] = np.random.uniform(0.25, 0.35, n_sac)

    # Blink: low velocity, low EAR
    X[n_fix+n_sac:, 0] = np.random.uniform(0, 20, n_blink)
    X[n_fix+n_sac:, 8] = np.random.uniform(0.05, 0.15, n_blink)

    y = np.array([0] * n_fix + [1] * n_sac + [2] * n_blink)

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print()

    # Run comparison
    comparison = MovementClassifierComparison(normalize=True)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=50, verbose=True)

    print("\n")
    comparison.print_results()

    wait_for_screenshot("Screenshot the movement classifier comparison table")

    comparison.print_per_class_results()

    wait_for_screenshot("Screenshot the per-class F1 scores table")

    # Show 3-class confusion matrix
    from evaluation import confusion_matrix, print_confusion_matrix

    print("\n--- 3-Class Confusion Matrix (Best Model) ---")
    best_result = comparison.get_best_model()
    best_model = best_result.model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, n_classes=3)
    print_confusion_matrix(cm, class_names=['Fixation', 'Saccade', 'Blink'])

    wait_for_screenshot("Screenshot the 3-class confusion matrix")


# =============================================================================
# SECTION 4: FEATURE SELECTION (PCA)
# =============================================================================
def demo_feature_selection():
    """Demo PCA feature selection."""
    section_header(4, "FEATURE SELECTION (PCA ANALYSIS)")

    print("This section demonstrates PCA for dimensionality reduction.")
    print("Goal: Find minimum components to retain 95% variance")
    print()

    from ml_from_scratch import PCA

    np.random.seed(42)

    # Generate 19-dimensional blink data
    X = np.random.randn(600, 19)

    # Run PCA
    pca = PCA(n_components=None)
    pca.fit(X)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print("=" * 50)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 50)
    print(f"{'Component':>10} {'Variance':>12} {'Cumulative':>12}")
    print("-" * 50)

    for i in range(min(10, len(explained_var))):
        print(f"{i+1:>10} {explained_var[i]:>12.4f} {cumulative_var[i]:>12.4f}")

    print("-" * 50)

    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    n_80 = np.argmax(cumulative_var >= 0.80) + 1

    print(f"\nComponents for 95% variance: {n_95}")
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 80% variance: {n_80}")
    print(f"\nDimensionality reduction: 19 -> {n_95} (95% variance)")
    print(f"Compression ratio: {19/n_95:.2f}x")

    wait_for_screenshot("Screenshot the PCA variance table")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        components = range(1, len(explained_var) + 1)
        ax.bar(components, explained_var, alpha=0.7, label='Individual', color='steelblue')
        ax.plot(components, cumulative_var, 'r-o', label='Cumulative', linewidth=2)
        ax.axhline(y=0.95, color='g', linestyle='--', label='95% threshold', linewidth=2)
        ax.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold', linewidth=2)

        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title('PCA - Explained Variance by Component', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/pca_demo.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to: results/pca_demo.png")
        plt.show()

        wait_for_screenshot("Screenshot the PCA plot window")

    except ImportError:
        print("\nMatplotlib not available - skipping plot")


# =============================================================================
# SECTION 5: EVALUATION METRICS
# =============================================================================
def demo_evaluation_metrics():
    """Demo evaluation metrics."""
    section_header(5, "EVALUATION METRICS DEMO")

    print("This section demonstrates all evaluation metrics.")
    print()

    from evaluation import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        classification_report, print_confusion_matrix,
        mean_squared_error, r2_score, mean_pixel_error
    )

    # Classification metrics demo
    print("=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    print()

    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    print("Confusion Matrix:")
    print_confusion_matrix(cm, class_names=['Class 0', 'Class 1', 'Class 2'])
    print()

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f} (macro)")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f} (macro)")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='macro'):.4f} (macro)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, class_names=['Class 0', 'Class 1', 'Class 2']))

    wait_for_screenshot("Screenshot the classification metrics")

    # Regression metrics demo
    print("\n" + "=" * 60)
    print("REGRESSION METRICS")
    print("=" * 60)

    y_true_reg = np.array([[100, 200], [150, 250], [200, 300], [250, 350], [300, 400]])
    y_pred_reg = np.array([[105, 195], [145, 255], [210, 290], [240, 360], [305, 395]])

    print("Gaze Estimation Example:")
    print(f"{'True (x,y)':>15} | {'Pred (x,y)':>15} | Error (px)")
    print("-" * 50)
    for i in range(5):
        error = np.sqrt((y_true_reg[i, 0] - y_pred_reg[i, 0])**2 +
                       (y_true_reg[i, 1] - y_pred_reg[i, 1])**2)
        print(f"({y_true_reg[i, 0]:3.0f}, {y_true_reg[i, 1]:3.0f})      | ({y_pred_reg[i, 0]:3.0f}, {y_pred_reg[i, 1]:3.0f})      | {error:6.2f}")

    mse = mean_squared_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    mpe = mean_pixel_error(y_true_reg, y_pred_reg)

    print(f"\nMSE:              {mse:.4f}")
    print(f"R2 Score:         {r2:.4f}")
    print(f"Mean Pixel Error: {mpe:.2f} px")

    wait_for_screenshot("Screenshot the regression metrics")


# =============================================================================
# SECTION 6: CROSS-VALIDATION
# =============================================================================
def demo_cross_validation():
    """Demo cross-validation."""
    section_header(6, "CROSS-VALIDATION DEMO")

    print("This section demonstrates stratified K-fold cross-validation.")
    print()

    from ml_from_scratch import RandomForestClassifier
    from evaluation import cross_val_score, stratified_k_fold

    np.random.seed(42)

    # Generate data
    X = np.random.randn(500, 19)
    y = np.array([0] * 250 + [1] * 250)
    idx = np.random.permutation(500)
    X, y = X[idx], y[idx]

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    print("=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)
    print(f"Model: Random Forest (n_estimators=50, max_depth=8)")
    print(f"Data: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y)}")
    print()

    # Run CV
    rf = RandomForestClassifier(n_estimators=50, max_depth=8)

    print("Running 5-fold cross-validation...")
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', stratify=True)

    print("\n" + "-" * 40)
    print(f"{'Fold':>6} | {'Accuracy':>10}")
    print("-" * 40)
    for i, score in enumerate(scores):
        print(f"{i+1:>6} | {score:>10.4f}")
    print("-" * 40)
    print(f"{'Mean':>6} | {np.mean(scores):>10.4f}")
    print(f"{'Std':>6} | {np.std(scores):>10.4f}")
    print("-" * 40)

    print(f"\nFinal Result: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"95% CI: [{np.mean(scores) - 1.96*np.std(scores):.4f}, {np.mean(scores) + 1.96*np.std(scores):.4f}]")

    wait_for_screenshot("Screenshot the cross-validation results")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    import argparse
    global INTERACTIVE

    parser = argparse.ArgumentParser(
        description="Black Box Demo - Eye Tracking ML Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sections:
  1 - Gaze Estimation (Regression Models)
  2 - Blink Detection (Binary Classifiers)
  3 - Eye Movement Classification (3-Class)
  4 - Feature Selection (PCA Analysis)
  5 - Evaluation Metrics Demo
  6 - Cross-Validation Demo
  all - Run all sections
        """
    )
    parser.add_argument("section", nargs="?", default="all",
                       help="Section to run (1-6 or 'all')")
    parser.add_argument("--no-pause", action="store_true",
                       help="Run without pausing for screenshots")

    args = parser.parse_args()

    if args.no_pause:
        INTERACTIVE = False

    print("\n" + "=" * 70)
    print("EYE TRACKING ML PROJECT - BLACK BOX DEMO")
    print("=" * 70)
    print("This script demonstrates all major functions for report documentation.")
    if INTERACTIVE:
        print("Press Enter after each section to continue after capturing screenshots.")
    else:
        print("Running in non-interactive mode (--no-pause)")
    print("=" * 70)

    sections = {
        '1': demo_gaze_estimation,
        '2': demo_blink_detection,
        '3': demo_movement_classification,
        '4': demo_feature_selection,
        '5': demo_evaluation_metrics,
        '6': demo_cross_validation,
    }

    if args.section.lower() == 'all':
        for num in ['1', '2', '3', '4', '5', '6']:
            sections[num]()
    elif args.section in sections:
        sections[args.section]()
    else:
        print(f"Unknown section: {args.section}")
        print("Use 1-6 or 'all'")
        return

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("All screenshots captured! Check your results/ folder for plots.")


if __name__ == "__main__":
    main()
