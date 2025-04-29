import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

true_ratings = torch.randint(1, 6, (1_000_000,), dtype=torch.float32, device=device)
pred_ratings = torch.clamp(true_ratings + torch.randn(1_000_000, device=device) * 0.8, 1, 5)

#Move to CPU for metric calculations if needed
true_ratings_cpu = true_ratings.cpu().numpy()
pred_ratings_cpu = pred_ratings.cpu().numpy()

mse = mean_squared_error(true_ratings_cpu, pred_ratings_cpu)
mae = mean_absolute_error(true_ratings_cpu, pred_ratings_cpu)

#Threshold for relevant items: rating >= 4
true_binary = (true_ratings_cpu >= 4).astype(int)
pred_binary = (pred_ratings_cpu >= 4).astype(int)

precision = precision_score(true_binary, pred_binary)
recall = recall_score(true_binary, pred_binary)
conf_matrix = confusion_matrix(true_binary, pred_binary)

print("Evaluation Metrics")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Precision (Threshold ≥4): {precision:.4f}")
print(f"Recall (Threshold ≥4): {recall:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

sample_indices = np.random.choice(len(true_ratings_cpu), 100000, replace=False)
df_eval = pd.DataFrame({
    "True Ratings": true_ratings_cpu[sample_indices],
    "Predicted Ratings": pred_ratings_cpu[sample_indices]
})

plt.figure(figsize=(10, 6))
sns.histplot(df_eval["True Ratings"], color="blue", label="True Ratings", kde=True, stat="density", bins=5)
sns.histplot(df_eval["Predicted Ratings"], color="orange", label="Predicted Ratings", kde=True, stat="density", bins=5)
plt.title("True vs Predicted Rating Distribution (Sampled)")
plt.xlabel("Rating")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
