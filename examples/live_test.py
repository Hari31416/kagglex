"""Diagnostic live test script for Kaggle remote execution."""

import json
import os
import sys
import time
from pathlib import Path

print("=" * 50)
print("Hello from remote Kaggle kernel execution!")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("=" * 50)

# Check PyTorch and GPU
cuda_available = False
device_names = []
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        count = torch.cuda.device_count()
        print(f"CUDA Device count: {count}")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            device_names.append(name)
            print(f"  GPU [{i}]: {name}")
except ImportError:
    print("PyTorch not installed in this environment.")

# Write output artifact to outputs/
outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)

test_results = {
    "timestamp": time.time(),
    "python_version": sys.version.split()[0],
    "cuda_available": cuda_available,
    "device_names": device_names,
    "status": "PASS",
}

results_file = outputs_dir / "live_test_results.json"
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=2)

print(f"Saved diagnostic output artifact to {results_file}")
print("Live test completed successfully.")
