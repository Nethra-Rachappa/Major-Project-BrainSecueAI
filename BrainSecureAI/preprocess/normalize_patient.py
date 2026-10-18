import nibabel as nib
import numpy as np
from pathlib import Path

# Patient folder
patient_path = Path(
    r"D:\BrainSecureAI\dataset\BraTS\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\BraTS-GLI-01535-000"
)

# Output folder
output_path = Path(r"D:\BrainSecureAI\outputs\preprocessed\BraTS-GLI-01535-000")
output_path.mkdir(parents=True, exist_ok=True)

files = {
    "T1c": "BraTS-GLI-01535-000-t1c.nii.gz",
    "T1n": "BraTS-GLI-01535-000-t1n.nii.gz",
    "T2f": "BraTS-GLI-01535-000-t2f.nii.gz",
    "T2w": "BraTS-GLI-01535-000-t2w.nii.gz",
}

for name, filename in files.items():

    file_path = patient_path / filename

    image = nib.load(file_path)
    data = image.get_fdata().astype(np.float32)

    # Use only non-zero voxels for normalization
    nonzero = data[data > 0]

    mean = np.mean(nonzero)
    std = np.std(nonzero)

    # Z-score normalization
    normalized = np.zeros_like(data)

    if std > 0:
        normalized[data > 0] = (data[data > 0] - mean) / std

    output_file = output_path / f"{name}_normalized.npy"
    np.save(output_file, normalized)

    print(f"\n{name}")
    print("Original mean:", mean)
    print("Original std:", std)
    print("Normalized minimum:", np.min(normalized))
    print("Normalized maximum:", np.max(normalized))

# Segmentation
seg_file = patient_path / "BraTS-GLI-01535-000-seg.nii.gz"

seg_image = nib.load(seg_file)
seg = seg_image.get_fdata().astype(np.uint8)

seg_output = output_path / "segmentation.npy"
np.save(seg_output, seg)

print("\nSegmentation")
print("Shape:", seg.shape)
print("Labels:", np.unique(seg))

print("\nPreprocessing completed successfully.")
print("Saved to:", output_path)