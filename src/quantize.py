import joblib
import numpy as np
import os
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load trained sklearn model
model = joblib.load("model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized params
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "unquant_params.joblib")

# Manual quantization to uint8
w_min, w_max = weights.min(), weights.max()
scale = 255 / (w_max - w_min)
zero_point = -w_min * scale

weights_q = np.round(weights * scale + zero_point).astype(np.uint8)
bias_q = np.round(bias * scale + zero_point).astype(np.uint8)

quant_params = {
    "weights_q": weights_q,
    "bias_q": bias_q,
    "scale": scale,
    "zero_point": zero_point,
}
joblib.dump(quant_params, "quant_params.joblib")

# Dequantize
weights_dq = (weights_q.astype(np.float32) - zero_point) / scale
bias_dq = (bias_q.astype(np.float32) - zero_point) / scale

# PyTorch Model using dequantized weights
class LinearModel(torch.nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.linear = torch.nn.Linear(len(weights), 1)
        self.linear.weight.data = torch.tensor([weights], dtype=torch.float32)
        self.linear.bias.data = torch.tensor([bias], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

# Load data
data = fetch_california_housing()
X = data.data
y = data.target

# Inference
model_torch = LinearModel(weights_dq, bias_dq)
model_torch.eval()
with torch.no_grad():
    y_pred = model_torch(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

# R² score
r2_quant = r2_score(y, y_pred)
print(f"R² Score (Quantized): {r2_quant:.4f}")

# Compare sizes
size_unquant = os.path.getsize("unquant_params.joblib") / 1024
size_quant = os.path.getsize("quant_params.joblib") / 1024
print(f"Unquantized Size: {size_unquant:.2f} KB")
print(f"Quantized Size:   {size_quant:.2f} KB")
