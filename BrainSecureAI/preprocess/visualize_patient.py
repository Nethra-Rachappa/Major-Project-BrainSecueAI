import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

patient_path = Path(
    r"D:\BrainSecureAI\dataset\BraTS\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\BraTS-GLI-01535-000"
)

t2f_file = patient_path / "BraTS-GLI-01535-000-t2f.nii.gz"
seg_file = patient_path / "BraTS-GLI-01535-000-seg.nii.gz"

t2f = nib.load(t2f_file).get_fdata()
seg = nib.load(seg_file).get_fdata()

# Find slices containing tumor
tumor_slices = np.where(np.any(seg > 0, axis=(0, 1)))[0]

# Select the middle tumor-containing slice
slice_index = tumor_slices[len(tumor_slices) // 2]

mri_slice = t2f[:, :, slice_index]
seg_slice = seg[:, :, slice_index]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.rot90(mri_slice), cmap="gray")
plt.title(f"T2-FLAIR - Slice {slice_index}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(mri_slice), cmap="gray")
plt.imshow(
    np.rot90(seg_slice),
    alpha=0.4,
    interpolation="nearest"
)
plt.title("MRI + Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()