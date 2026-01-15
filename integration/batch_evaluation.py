"""
Batch evaluation for Eye Tracking ML Project.

Runs complete evaluation pipeline:
- Gaze estimation model comparison
- Blink detection classifier comparison
- Eye movement classification comparison
- Feature selection comparison
- Cross-validation results
- Generates tables, plots, and final report

All results are saved to the results/ directory for documentation.
"""

import numpy as np
import os
import sys
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.dataset.dataset_manager import DatasetManager

# Import ML modules
from ml_from_scratch import (
    LinearRegression, RidgeRegression, PolynomialRegression,
    DecisionTreeClassifier, RandomForestClassifier, XGBoostClassifier,
    KNeighborsClassifier, SVC, GaussianNB, NeuralNetwork,
    PCA, GeneticAlgorithmFeatureSelector, LinearDiscriminantAnalysis
)

# Import evaluation metrics
from evaluation import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, roc_auc_score, multi_class_roc_auc,
    mean_squared_error, mean_absolute_error, r2_score, mean_pixel_error,
    cross_val_score, stratified_k_fold,
    plot_confusion_matrix, plot_roc_curve, plot_multi_class_roc
)

# Import comparison utilities
from gaze_estimation import GazeModelComparison
from classification import ClassifierComparison, MovementClassifierComparison
from feature_selection import FeatureSelectionComparison


# Results directory structure
RESULTS_DIR = "results"
PHASE_DIRS = {
    "phase1": "phase1",
    "phase2": "phase2",
    "phase3_gaze": "phase3_gaze",
    "phase4_blink": "phase4_blink",
    "phase5_movement": "phase5_movement",
    "phase6_features": "phase6_features",
    "phase7_evaluation": "phase7_evaluation",
    "phase8_integration": "phase8_integration",
}


class ResultsDocumenter:
    """
    Document and export all evaluation results.

    Provides utilities for:
    - Saving markdown tables
    - Saving CSV files
    - Generating plots
    - Compiling final report
    """

    def __init__(self, results_dir: str = RESULTS_DIR):
        """Initialize documenter and create directory structure."""
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results: Dict[str, Dict] = {}

        # Create directory structure
        self._create_directories()

    def _create_directories(self):
        """Create results directory structure."""
        for phase, dirname in PHASE_DIRS.items():
            phase_dir = os.path.join(self.results_dir, dirname)
            os.makedirs(phase_dir, exist_ok=True)
            os.makedirs(os.path.join(phase_dir, "plots"), exist_ok=True)

    def add_result(self, phase: str, name: str, data: Any):
        """Add a result to the collection."""
        if phase not in self.results:
            self.results[phase] = {}
        self.results[phase][name] = data

    def save_table_md(self, data: List[List], headers: List[str],
                      filepath: str, title: str = None):
        """Save data as markdown table."""
        with open(filepath, 'w', encoding='utf-8') as f:
            if title:
                f.write(f"# {title}\n\n")
                f.write(f"Generated: {self.timestamp}\n\n")

            # Header
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

            # Rows
            for row in data:
                f.write("| " + " | ".join(str(x) for x in row) + " |\n")

        print(f"  Saved: {filepath}")

    def save_table_csv(self, data: List[List], headers: List[str], filepath: str):
        """Save data as CSV for Excel."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"  Saved: {filepath}")

    def save_json(self, data: Dict, filepath: str):
        """Save data as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {filepath}")

    def save_summary(self, content: str, filepath: str, title: str = None):
        """Save text summary to markdown file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            if title:
                f.write(f"# {title}\n\n")
                f.write(f"Generated: {self.timestamp}\n\n")
            f.write(content)
        print(f"  Saved: {filepath}")

    def compile_final_report(self, filepath: str):
        """Compile all results into a single markdown file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Eye Tracking ML Project - Complete Results\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            f.write("---\n\n")

            for phase, phase_results in self.results.items():
                f.write(f"## {phase}\n\n")

                for name, data in phase_results.items():
                    f.write(f"### {name}\n\n")

                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"- **{key}**: {value}\n")
                    elif isinstance(data, str):
                        f.write(f"{data}\n")
                    else:
                        f.write(f"{data}\n")

                    f.write("\n")

                f.write("---\n\n")

        print(f"  Compiled: {filepath}")


class BatchEvaluation:
    """
    Complete batch evaluation of all ML models.

    Evaluates:
    1. Gaze estimation (regression models)
    2. Blink detection (binary classifiers)
    3. Eye movement classification (multi-class)
    4. Feature selection methods
    5. Cross-validation results
    """

    def __init__(self, data_dir: str = "data", results_dir: str = RESULTS_DIR):
        """Initialize batch evaluation."""
        self.data_manager = DatasetManager(data_dir=data_dir)
        self.documenter = ResultsDocumenter(results_dir=results_dir)
        self.results_dir = results_dir

        # Store evaluation results
        self.gaze_results: Optional[Dict] = None
        self.blink_results: Optional[Dict] = None
        self.movement_results: Optional[Dict] = None
        self.feature_selection_results: Optional[Dict] = None

    def run_all(self, use_synthetic: bool = True, verbose: bool = True):
        """
        Run complete evaluation pipeline.

        Args:
            use_synthetic: If True, use synthetic data when real data unavailable
            verbose: If True, print progress
        """
        print("\n" + "=" * 70)
        print("EYE TRACKING ML PROJECT - BATCH EVALUATION")
        print("=" * 70)
        print(f"Timestamp: {self.documenter.timestamp}")
        print("=" * 70)

        start_time = time.time()

        # Run each phase
        self._evaluate_gaze(use_synthetic, verbose)
        self._evaluate_blink(use_synthetic, verbose)
        self._evaluate_movement(use_synthetic, verbose)
        self._evaluate_feature_selection(use_synthetic, verbose)
        self._evaluate_cross_validation(use_synthetic, verbose)

        # Generate final report
        self._generate_final_report()

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"EVALUATION COMPLETE - Total time: {elapsed:.1f} seconds")
        print("=" * 70)
        print(f"\nResults saved to: {self.results_dir}/")
        print("Key files:")
        print("  - FINAL_REPORT_DATA.md (combined results)")
        print("  - all_results.json (programmatic access)")
        print("  - phase*/classifier_comparison.md (comparison tables)")
        print("  - phase*/plots/*.png (visualizations)")

    def _evaluate_gaze(self, use_synthetic: bool, verbose: bool):
        """Evaluate gaze estimation models."""
        print("\n" + "-" * 60)
        print("PHASE 3: GAZE ESTIMATION")
        print("-" * 60)

        # Load or generate data
        try:
            X, y, info = self.data_manager.load_dataset("gaze_calibration")
            print(f"Loaded gaze dataset: {len(X)} samples")
        except Exception:
            if use_synthetic:
                print("Using synthetic gaze data...")
                X, y = self._generate_synthetic_gaze_data()
            else:
                print("No gaze data available")
                return

        # Split data
        X_train, X_test, y_train, y_test = self.data_manager.train_test_split(
            X, y, test_size=0.2, stratify=False
        )

        # Run comparison
        comparison = GazeModelComparison(normalize=True)
        comparison.load_data(X_train, y_train, X_test, y_test)
        comparison.run_comparison(include_nn=True, nn_epochs=100, verbose=verbose)

        if verbose:
            comparison.print_results()

        # Save results
        self.gaze_results = comparison.to_dict()
        self._save_gaze_results(comparison)

    def _evaluate_blink(self, use_synthetic: bool, verbose: bool):
        """Evaluate blink detection classifiers."""
        print("\n" + "-" * 60)
        print("PHASE 4: BLINK DETECTION")
        print("-" * 60)

        # Load or generate data
        try:
            X, y, info = self.data_manager.load_dataset("blink_balanced")
            print(f"Loaded blink dataset: {len(X)} samples")
        except Exception:
            if use_synthetic:
                print("Using synthetic blink data...")
                X, y = self._generate_synthetic_blink_data()
            else:
                print("No blink data available")
                return

        # Split data
        X_train, X_test, y_train, y_test = self.data_manager.train_test_split(
            X, y, test_size=0.2, stratify=True
        )

        # Run comparison
        comparison = ClassifierComparison(normalize=True, task='blink')
        comparison.load_data(X_train, y_train, X_test, y_test)
        comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=50, verbose=verbose)

        if verbose:
            comparison.print_results()

        # Save results
        self.blink_results = comparison.to_dict()
        self._save_blink_results(comparison, X_test, y_test)

    def _evaluate_movement(self, use_synthetic: bool, verbose: bool):
        """Evaluate eye movement classifiers."""
        print("\n" + "-" * 60)
        print("PHASE 5: EYE MOVEMENT CLASSIFICATION")
        print("-" * 60)

        # Load or generate data
        try:
            X, y, info = self.data_manager.load_dataset("movement")
            print(f"Loaded movement dataset: {len(X)} samples")
        except Exception:
            if use_synthetic:
                print("Using synthetic movement data...")
                X, y = self._generate_synthetic_movement_data()
            else:
                print("No movement data available")
                return

        # Split data
        X_train, X_test, y_train, y_test = self.data_manager.train_test_split(
            X, y, test_size=0.2, stratify=True
        )

        # Run comparison
        comparison = MovementClassifierComparison(normalize=True)
        comparison.load_data(X_train, y_train, X_test, y_test)
        comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=50, verbose=verbose)

        if verbose:
            comparison.print_results()
            comparison.print_per_class_results()

        # Save results
        self.movement_results = comparison.to_dict()
        self._save_movement_results(comparison, X_test, y_test)

    def _evaluate_feature_selection(self, use_synthetic: bool, verbose: bool):
        """Evaluate feature selection methods."""
        print("\n" + "-" * 60)
        print("PHASE 6: FEATURE SELECTION")
        print("-" * 60)

        # Use blink data for feature selection evaluation
        try:
            X, y, info = self.data_manager.load_dataset("blink_balanced")
        except Exception:
            if use_synthetic:
                X, y = self._generate_synthetic_blink_data()
            else:
                print("No data available for feature selection")
                return

        # Run PCA analysis
        print("Running PCA analysis...")
        pca = PCA(n_components=None)
        pca.fit(X)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        n_95 = np.argmax(cumulative_var >= 0.95) + 1
        n_90 = np.argmax(cumulative_var >= 0.90) + 1
        n_80 = np.argmax(cumulative_var >= 0.80) + 1

        print(f"  95% variance: {n_95} components")
        print(f"  90% variance: {n_90} components")
        print(f"  80% variance: {n_80} components")

        # Save results
        self.feature_selection_results = {
            "pca": {
                "n_95_variance": n_95,
                "n_90_variance": n_90,
                "n_80_variance": n_80,
                "explained_variance": explained_var.tolist()[:10],
                "cumulative_variance": cumulative_var.tolist()[:10]
            }
        }
        self._save_feature_selection_results(pca, explained_var, cumulative_var)

    def _evaluate_cross_validation(self, use_synthetic: bool, verbose: bool):
        """Run cross-validation on best models."""
        print("\n" + "-" * 60)
        print("PHASE 7: CROSS-VALIDATION")
        print("-" * 60)

        # Use blink data
        try:
            X, y, _ = self.data_manager.load_dataset("blink_balanced")
        except Exception:
            if use_synthetic:
                X, y = self._generate_synthetic_blink_data()
            else:
                print("No data for cross-validation")
                return

        # Normalize data
        X_train, _, stats = self.data_manager.normalize(X)

        # Cross-validate Random Forest
        print("Cross-validating Random Forest (5-fold)...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=8)
        cv_scores = cross_val_score(rf, X_train, y, cv=5, scoring='accuracy', stratify=True)

        print(f"  Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Save results
        cv_results = {
            "model": "Random Forest",
            "cv_folds": 5,
            "scores": cv_scores.tolist(),
            "mean_score": float(np.mean(cv_scores)),
            "std_score": float(np.std(cv_scores))
        }

        self.documenter.add_result("Phase 7: Cross-Validation", "Results", cv_results)

        filepath = os.path.join(self.results_dir, "phase7_evaluation", "cross_validation.md")
        content = f"""## Cross-Validation Results

**Model**: Random Forest (n_estimators=50, max_depth=8)
**CV Folds**: 5 (Stratified)

### Fold Scores
| Fold | Accuracy |
| --- | --- |
"""
        for i, score in enumerate(cv_scores):
            content += f"| {i+1} | {score:.4f} |\n"

        content += f"""
### Summary
- **Mean Accuracy**: {np.mean(cv_scores):.4f}
- **Std Dev**: {np.std(cv_scores):.4f}
- **95% CI**: [{np.mean(cv_scores) - 1.96*np.std(cv_scores):.4f}, {np.mean(cv_scores) + 1.96*np.std(cv_scores):.4f}]
"""
        self.documenter.save_summary(content, filepath, "Cross-Validation Results")

    def _generate_synthetic_gaze_data(self, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic gaze estimation data."""
        np.random.seed(42)
        X = np.random.randn(n_samples, 10)

        # Non-linear gaze mapping
        y_x = 500 + 300 * X[:, 0] + 250 * X[:, 2] + 50 * X[:, 0] * X[:, 2]
        y_y = 400 + 200 * X[:, 1] + 180 * X[:, 3] + 40 * X[:, 1] * X[:, 3]
        y = np.column_stack([y_x, y_y]) + np.random.randn(n_samples, 2) * 25

        return X, y

    def _generate_synthetic_blink_data(self, n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic blink detection data."""
        np.random.seed(42)
        X = np.random.randn(n_samples, 19)

        # EAR-based features (first feature is current EAR)
        X[:n_samples//2, 0] = np.random.uniform(0.25, 0.35, n_samples//2)  # Open
        X[n_samples//2:, 0] = np.random.uniform(0.05, 0.15, n_samples//2)  # Blink
        y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

        # Shuffle
        idx = np.random.permutation(n_samples)
        return X[idx], y[idx]

    def _generate_synthetic_movement_data(self, n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic movement classification data."""
        np.random.seed(42)
        X = np.random.randn(n_samples, 14)

        # Class-specific features
        n_fix = n_samples // 2
        n_sac = n_samples // 3
        n_blink = n_samples - n_fix - n_sac

        # Fixation: low velocity
        X[:n_fix, 0] = np.random.uniform(0, 10, n_fix)
        X[:n_fix, 8] = np.random.uniform(0.25, 0.35, n_fix)  # EAR

        # Saccade: high velocity
        X[n_fix:n_fix+n_sac, 0] = np.random.uniform(50, 200, n_sac)
        X[n_fix:n_fix+n_sac, 8] = np.random.uniform(0.25, 0.35, n_sac)

        # Blink: low EAR
        X[n_fix+n_sac:, 0] = np.random.uniform(0, 20, n_blink)
        X[n_fix+n_sac:, 8] = np.random.uniform(0.05, 0.15, n_blink)

        y = np.array([0] * n_fix + [1] * n_sac + [2] * n_blink)

        idx = np.random.permutation(n_samples)
        return X[idx], y[idx]

    def _save_gaze_results(self, comparison: GazeModelComparison):
        """Save gaze estimation results."""
        results = comparison.to_dict()

        # Prepare table data
        table_data = []
        for r in results:
            table_data.append([
                r['name'],
                f"{r['train_error']:.1f}",
                f"{r['test_error']:.1f}",
                f"{r['test_r2']:.4f}",
                f"{r['train_time']:.3f}"
            ])

        # Sort by test error
        table_data.sort(key=lambda x: float(x[2]))

        headers = ["Model", "Train Error (px)", "Test Error (px)", "R² Score", "Time (s)"]

        # Save files
        base_path = os.path.join(self.results_dir, "phase3_gaze")
        self.documenter.save_table_md(
            table_data, headers,
            os.path.join(base_path, "regression_comparison.md"),
            "Gaze Estimation - Model Comparison"
        )
        self.documenter.save_table_csv(
            table_data, headers,
            os.path.join(base_path, "regression_comparison.csv")
        )

        # Best model summary
        best = table_data[0]
        summary = f"""**Best Model**: {best[0]}
**Test Error**: {best[2]} pixels
**R² Score**: {best[3]}
**Training Time**: {best[4]} seconds
"""
        self.documenter.save_summary(
            summary,
            os.path.join(base_path, "best_model_summary.md"),
            "Best Gaze Estimation Model"
        )

        self.documenter.add_result(
            "Phase 3: Gaze Estimation", "Best Model",
            {"name": best[0], "test_error": best[2], "r2": best[3]}
        )

    def _save_blink_results(self, comparison: ClassifierComparison,
                            X_test: np.ndarray, y_test: np.ndarray):
        """Save blink detection results."""
        # to_dict() returns List[Dict] with name, train_accuracy, test_accuracy, precision, recall, f1_score, train_time
        results_list = comparison.to_dict()

        # Prepare table data
        table_data = []
        for r in results_list:
            table_data.append([
                r['name'],
                f"{r['train_accuracy']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1_score']:.4f}",
                f"{r['train_time']:.3f}"
            ])

        # Sort by test accuracy
        table_data.sort(key=lambda x: float(x[2]), reverse=True)

        headers = ["Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1", "Time (s)"]

        base_path = os.path.join(self.results_dir, "phase4_blink")

        self.documenter.save_table_md(
            table_data, headers,
            os.path.join(base_path, "classifier_comparison.md"),
            "Blink Detection - Classifier Comparison"
        )
        self.documenter.save_table_csv(
            table_data, headers,
            os.path.join(base_path, "classifier_comparison.csv")
        )

        # Generate confusion matrices for top 3 models using comparison.results (ClassifierResult objects)
        top_models = [row[0] for row in table_data[:3]]
        results_dict = {r.name: r for r in comparison.results}
        for model_name in top_models:
            if model_name in results_dict:
                result = results_dict[model_name]
                cm = result.confusion_matrix
                safe_name = model_name.replace(" ", "_").lower()
                plot_confusion_matrix(
                    cm, class_names=['No Blink', 'Blink'],
                    title=f"Confusion Matrix - {model_name}",
                    save_path=os.path.join(base_path, "plots", f"cm_{safe_name}.png")
                )

        # Best model summary
        best = table_data[0]
        self.documenter.add_result(
            "Phase 4: Blink Detection", "Best Model",
            {"name": best[0], "accuracy": best[2], "f1": best[5]}
        )

    def _save_movement_results(self, comparison: MovementClassifierComparison,
                                X_test: np.ndarray, y_test: np.ndarray):
        """Save movement classification results."""
        # to_dict() returns {'models': [{'name': ..., 'test_accuracy': ..., 'overall_f1': ..., 'per_class': [...], ...}]}
        results_dict = comparison.to_dict()
        model_results = results_dict['models']

        # Prepare table data
        table_data = []
        for r in model_results:
            # Extract per-class F1 scores from per_class list
            per_class = {pc['class']: pc['f1'] for pc in r.get('per_class', [])}
            fixation_f1 = per_class.get('Fixation', 0)
            saccade_f1 = per_class.get('Saccade', 0)
            blink_f1 = per_class.get('Blink', 0)

            table_data.append([
                r['name'],
                f"{r['test_accuracy']:.4f}",
                f"{fixation_f1:.4f}",
                f"{saccade_f1:.4f}",
                f"{blink_f1:.4f}",
                f"{r['overall_f1']:.4f}"
            ])

        table_data.sort(key=lambda x: float(x[1]), reverse=True)

        headers = ["Model", "Accuracy", "Fixation F1", "Saccade F1", "Blink F1", "Macro F1"]

        base_path = os.path.join(self.results_dir, "phase5_movement")

        self.documenter.save_table_md(
            table_data, headers,
            os.path.join(base_path, "classifier_comparison.md"),
            "Eye Movement Classification - Model Comparison"
        )
        self.documenter.save_table_csv(
            table_data, headers,
            os.path.join(base_path, "classifier_comparison.csv")
        )

        # Generate 3-class confusion matrix for best model using comparison.results
        best_name = table_data[0][0]
        results_by_name = {r.name: r for r in comparison.results}
        if best_name in results_by_name:
            result = results_by_name[best_name]
            cm = result.confusion_matrix
            plot_confusion_matrix(
                cm, class_names=['Fixation', 'Saccade', 'Blink'],
                title=f"Movement Classification - {best_name}",
                save_path=os.path.join(base_path, "plots", "confusion_matrix_3class.png")
            )

        # Best model summary
        best = table_data[0]
        self.documenter.add_result(
            "Phase 5: Movement Classification", "Best Model",
            {"name": best[0], "accuracy": best[1], "macro_f1": best[5]}
        )

    def _save_feature_selection_results(self, pca: PCA,
                                         explained_var: np.ndarray,
                                         cumulative_var: np.ndarray):
        """Save feature selection results."""
        base_path = os.path.join(self.results_dir, "phase6_features")

        # PCA table
        pca_data = []
        for i, (var, cum) in enumerate(zip(explained_var[:10], cumulative_var[:10])):
            pca_data.append([i + 1, f"{var:.4f}", f"{cum:.4f}"])

        self.documenter.save_table_md(
            pca_data,
            ["Component", "Variance Ratio", "Cumulative"],
            os.path.join(base_path, "pca_analysis.md"),
            "PCA Analysis"
        )

        # Plot explained variance
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))

            components = range(1, len(explained_var) + 1)
            ax.bar(components, explained_var, alpha=0.7, label='Individual')
            ax.plot(components, cumulative_var, 'r-o', label='Cumulative')
            ax.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
            ax.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')

            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA - Explained Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig(os.path.join(base_path, "plots", "explained_variance.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {base_path}/plots/explained_variance.png")
        except ImportError:
            print("  Matplotlib not available for PCA plot")

        # Summary
        n_95 = np.argmax(cumulative_var >= 0.95) + 1
        self.documenter.add_result(
            "Phase 6: Feature Selection", "PCA Results",
            {"n_components_95_var": n_95}
        )

    def _generate_final_report(self):
        """Generate the final compiled report."""
        print("\n" + "-" * 60)
        print("GENERATING FINAL REPORT")
        print("-" * 60)

        # Compile final report
        self.documenter.compile_final_report(
            os.path.join(self.results_dir, "FINAL_REPORT_DATA.md")
        )

        # Save all results as JSON
        self.documenter.save_json(
            self.documenter.results,
            os.path.join(self.results_dir, "all_results.json")
        )


def run_full_evaluation(data_dir: str = "data", results_dir: str = "results",
                        use_synthetic: bool = True, verbose: bool = True):
    """
    Convenience function to run the full evaluation pipeline.

    Args:
        data_dir: Directory containing datasets
        results_dir: Directory for output results
        use_synthetic: If True, use synthetic data when real data unavailable
        verbose: If True, print detailed progress

    Returns:
        BatchEvaluation instance with results
    """
    evaluation = BatchEvaluation(data_dir=data_dir, results_dir=results_dir)
    evaluation.run_all(use_synthetic=use_synthetic, verbose=verbose)
    return evaluation


# =============================================================================
# Test functions
# =============================================================================

def test_batch_evaluation():
    """Test batch evaluation with synthetic data."""
    print("Running batch evaluation test...")
    evaluation = run_full_evaluation(use_synthetic=True, verbose=True)
    return evaluation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eye Tracking ML Batch Evaluation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data if real data unavailable")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    run_full_evaluation(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_synthetic=args.synthetic,
        verbose=not args.quiet
    )
