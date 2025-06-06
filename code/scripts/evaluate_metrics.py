from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Generate synthetic data
np.random.seed(42)  # for reproducibility

# Create ground truth (gt) with text labels
gt = ['healthy', 'unhealthy', 'unhealthy', 'healthy', 'unhealthy', 
    'healthy', 'unhealthy', 'unhealthy', 'healthy', 'healthy']

# Create predictions (pred) with text labels
pred = ['healthy', 'unhealthy', 'unhealthy', 'healthy', 'healthy', 
      'healthy', 'unhealthy', 'healthy', 'healthy', 'unhealthy']

# Calculate metrics
accuracy = accuracy_score(gt, pred)
f1 = f1_score(gt, pred, pos_label='unhealthy')

# Print results
print(f"Ground Truth: {gt}")
print(f"Predictions: {pred}")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")