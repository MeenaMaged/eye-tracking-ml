# Gaze Estimation - Model Comparison

Generated: 2026-01-15 22:18:19

| Model | Train Error (px) | Test Error (px) | R² Score | Time (s) |
| --- | --- | --- | --- | --- |
| Polynomial (deg=2) | 27.6 | 34.8 | 0.9932 | 0.003 |
| Linear Regression | 56.3 | 65.5 | 0.9721 | 0.000 |
| Ridge (a=1.0) | 56.3 | 65.8 | 0.9719 | 0.000 |
| Polynomial (deg=3) | 12.9 | 134.7 | 0.8569 | 0.023 |
| ANN (64-32) | 417.2 | 423.5 | -0.0603 | 0.118 |
