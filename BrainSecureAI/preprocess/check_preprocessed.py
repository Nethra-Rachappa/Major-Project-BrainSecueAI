import numpy as np
from pathlib import Path

patient_path = Path(
    r"D:\BrainSecureAI\outputs\preprocessed"
    r"\BraTS-GLI-01535-000"
)

files = [
    "T1c_normalized.npy",
    "T1n_normalized.npy",
    "T2f_normalized.npy",
    "T2w_normalized.npy",
    "segmentation.npy"
]

for filename in files:

    file_path = patient_path / filename

    data = np.load(file_path)

    print("\n", filename)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))

print("\nAll preprocessed files loaded successfully.")