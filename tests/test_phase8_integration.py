"""
Phase 8 Integration Tests.

Tests for:
- Live demo functionality (without webcam)
- Batch evaluation pipeline
- Results documentation generation
"""

import numpy as np
import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_integration_imports():
    """Test that integration module imports correctly."""
    print("=" * 60)
    print("TEST: Integration Module Imports")
    print("=" * 60)

    try:
        from integration import (
            LiveDemo,
            run_live_demo,
            BatchEvaluation,
            ResultsDocumenter,
            run_full_evaluation,
        )
        print("  LiveDemo")
        print("  run_live_demo")
        print("  BatchEvaluation")
        print("  ResultsDocumenter")
        print("  run_full_evaluation")
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def test_results_documenter():
    """Test ResultsDocumenter class."""
    print("\n" + "=" * 60)
    print("TEST: ResultsDocumenter")
    print("=" * 60)

    from integration.batch_evaluation import ResultsDocumenter

    # Create temp directory for test results
    temp_dir = tempfile.mkdtemp(prefix="eye_tracking_test_")

    try:
        doc = ResultsDocumenter(results_dir=temp_dir)

        # Test add_result
        doc.add_result("Phase 1", "Test Metric", {"accuracy": 0.95, "f1": 0.92})
        print("  add_result: OK")

        # Test save_table_md
        table_data = [
            ["Model A", "0.95", "0.92"],
            ["Model B", "0.93", "0.90"],
        ]
        headers = ["Model", "Accuracy", "F1"]
        doc.save_table_md(
            table_data, headers,
            os.path.join(temp_dir, "test_table.md"),
            "Test Results"
        )

        # Verify file was created
        assert os.path.exists(os.path.join(temp_dir, "test_table.md"))
        print("  save_table_md: OK")

        # Test save_json
        doc.save_json(
            {"test": "data", "values": [1, 2, 3]},
            os.path.join(temp_dir, "test.json")
        )
        assert os.path.exists(os.path.join(temp_dir, "test.json"))
        print("  save_json: OK")

        # Test compile_final_report
        doc.compile_final_report(os.path.join(temp_dir, "final_report.md"))
        assert os.path.exists(os.path.join(temp_dir, "final_report.md"))
        print("  compile_final_report: OK")

        print("\nResultsDocumenter: All tests passed!")
        return True

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_batch_evaluation_synthetic():
    """Test BatchEvaluation with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: BatchEvaluation (Synthetic Data)")
    print("=" * 60)

    from integration.batch_evaluation import BatchEvaluation

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="eye_tracking_eval_")

    try:
        # Create evaluation instance
        eval_instance = BatchEvaluation(
            data_dir=os.path.join(temp_dir, "data"),
            results_dir=os.path.join(temp_dir, "results")
        )

        # Test synthetic data generation
        X_gaze, y_gaze = eval_instance._generate_synthetic_gaze_data(n_samples=100)
        assert X_gaze.shape == (100, 10), f"Gaze X shape: {X_gaze.shape}"
        assert y_gaze.shape == (100, 2), f"Gaze y shape: {y_gaze.shape}"
        print("  Synthetic gaze data: OK")

        X_blink, y_blink = eval_instance._generate_synthetic_blink_data(n_samples=100)
        assert X_blink.shape == (100, 19), f"Blink X shape: {X_blink.shape}"
        assert y_blink.shape == (100,), f"Blink y shape: {y_blink.shape}"
        assert set(np.unique(y_blink)) == {0, 1}, "Blink labels should be 0 and 1"
        print("  Synthetic blink data: OK")

        X_move, y_move = eval_instance._generate_synthetic_movement_data(n_samples=100)
        assert X_move.shape == (100, 14), f"Movement X shape: {X_move.shape}"
        assert y_move.shape == (100,), f"Movement y shape: {y_move.shape}"
        assert set(np.unique(y_move)) == {0, 1, 2}, "Movement labels should be 0, 1, 2"
        print("  Synthetic movement data: OK")

        print("\nBatchEvaluation data generation: All tests passed!")
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_live_demo_stats():
    """Test LiveDemo statistics tracking."""
    print("\n" + "=" * 60)
    print("TEST: LiveDemo Statistics")
    print("=" * 60)

    from integration.live_demo import DemoStats

    stats = DemoStats(start_time=0)
    stats.frames_processed = 100
    stats.faces_detected = 95
    stats.blinks_detected = 5
    stats.fixations = 80
    stats.saccades = 10

    assert stats.detection_rate == 0.95, f"Detection rate: {stats.detection_rate}"
    print(f"  Detection rate: {stats.detection_rate * 100:.1f}%")

    print("\nLiveDemo stats: All tests passed!")
    return True


def test_full_evaluation_quick():
    """Run a quick full evaluation test."""
    print("\n" + "=" * 60)
    print("TEST: Full Evaluation Pipeline (Quick)")
    print("=" * 60)
    print("This test runs a minimal evaluation with synthetic data.")

    from integration.batch_evaluation import BatchEvaluation

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="eye_tracking_full_")

    try:
        eval_instance = BatchEvaluation(
            data_dir=os.path.join(temp_dir, "data"),
            results_dir=os.path.join(temp_dir, "results")
        )

        # Test individual phases
        print("\nTesting gaze evaluation...")
        eval_instance._evaluate_gaze(use_synthetic=True, verbose=False)
        print("  Gaze evaluation: OK")

        print("\nTesting blink evaluation...")
        eval_instance._evaluate_blink(use_synthetic=True, verbose=False)
        print("  Blink evaluation: OK")

        print("\nTesting movement evaluation...")
        eval_instance._evaluate_movement(use_synthetic=True, verbose=False)
        print("  Movement evaluation: OK")

        print("\nTesting feature selection...")
        eval_instance._evaluate_feature_selection(use_synthetic=True, verbose=False)
        print("  Feature selection: OK")

        # Generate report
        print("\nGenerating final report...")
        eval_instance._generate_final_report()

        # Verify output files
        results_dir = os.path.join(temp_dir, "results")
        assert os.path.exists(os.path.join(results_dir, "FINAL_REPORT_DATA.md"))
        assert os.path.exists(os.path.join(results_dir, "all_results.json"))
        print("  Final report generated: OK")

        print("\nFull evaluation: All tests passed!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_all_ml_models_available():
    """Test that all ML models can be imported."""
    print("\n" + "=" * 60)
    print("TEST: ML Models Availability")
    print("=" * 60)

    models_ok = True

    # Regressors
    try:
        from ml_from_scratch import LinearRegression, RidgeRegression, PolynomialRegression
        print("  Regressors (Linear, Ridge, Polynomial): OK")
    except ImportError as e:
        print(f"  Regressors: FAILED - {e}")
        models_ok = False

    # Classifiers
    try:
        from ml_from_scratch import (
            DecisionTreeClassifier, RandomForestClassifier, XGBoostClassifier,
            KNeighborsClassifier, SVC, GaussianNB
        )
        print("  Classifiers (DT, RF, XGB, KNN, SVM, NB): OK")
    except ImportError as e:
        print(f"  Classifiers: FAILED - {e}")
        models_ok = False

    # Neural Network
    try:
        from ml_from_scratch import NeuralNetwork
        print("  Neural Network: OK")
    except ImportError as e:
        print(f"  Neural Network: FAILED - {e}")
        models_ok = False

    # Feature Selection
    try:
        from ml_from_scratch import PCA, GeneticAlgorithmFeatureSelector, LinearDiscriminantAnalysis
        print("  Feature Selection (PCA, GA, LDA): OK")
    except ImportError as e:
        print(f"  Feature Selection: FAILED - {e}")
        models_ok = False

    # Evaluation
    try:
        from evaluation import (
            confusion_matrix, accuracy_score, f1_score,
            roc_curve, roc_auc_score,
            mean_squared_error, r2_score,
            cross_val_score
        )
        print("  Evaluation Metrics: OK")
    except ImportError as e:
        print(f"  Evaluation: FAILED - {e}")
        models_ok = False

    if models_ok:
        print("\nAll ML models available!")
    else:
        print("\nSome models failed to import")

    return models_ok


def run_all_phase8_tests():
    """Run all Phase 8 tests."""
    print("\n" + "=" * 70)
    print("PHASE 8 INTEGRATION TESTS")
    print("=" * 70)

    results = {}

    # Test imports
    results["Imports"] = test_integration_imports()

    # Test ML models
    results["ML Models"] = test_all_ml_models_available()

    # Test ResultsDocumenter
    results["ResultsDocumenter"] = test_results_documenter()

    # Test BatchEvaluation synthetic data
    results["Synthetic Data"] = test_batch_evaluation_synthetic()

    # Test LiveDemo stats
    results["LiveDemo Stats"] = test_live_demo_stats()

    # Test full evaluation (quick)
    results["Full Evaluation"] = test_full_evaluation_quick()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "OK" if result else "XX"
        print(f"  [{icon}] {test_name}: {status}")
        if result:
            passed += 1

    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 8 Integration Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")

    args = parser.parse_args()

    if args.quick:
        test_integration_imports()
        test_all_ml_models_available()
    else:
        success = run_all_phase8_tests()
        sys.exit(0 if success else 1)
