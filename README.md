# Deepfake-Detection-in-video-using-integrity-verification-method

## Project Overview
The approach analyzes abnormal eye blinking behavior and facial alignment errors to
identify manipulated video content. These features are passed to an SVM classifier,
and the final decision is interpreted using rule-based thresholds.

- Feature extraction based on:
  - Eye blink rate
  - Alignment error (AE)
- Classification using Support Vector Machine (SVM)
- Post-classification decision rules based on experimental thresholds

The model achieves balanced precision, recall, and F1-score with an accuracy of
approximately 80–85%, consistent with the evaluation report
