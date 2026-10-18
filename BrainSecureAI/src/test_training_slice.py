
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.dataset_loader import BraTSDataset

DATASET_PATH = (
    r"D:\BrainSecureAI\dataset\BraTS\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

dataset = BraTSDataset(DATASET_PATH)

# Load first patient
image, segmentation = dataset[0]

# Identify axial slices containing tumor tissue
tumor_slices = torch.where(
    torch.any(segmentation > 0, dim=(0, 1))
)[0]

if len(tumor_slices) == 0:
    raise ValueError("No tumor-containing slices found")

# Select middle tumor-containing slice
slice_index = tumor_slices[len(tumor_slices) // 2].item()

# Extract all four MRI channels
image_slice = image[:, :, :, slice_index]

# Extract segmentation
seg_slice = segmentation[:, :, slice_index]

print("Selected slice:", slice_index)
print("MRI slice shape:", image_slice.shape)
print("Segmentation shape:", seg_slice.shape)
print("Segmentation labels:", torch.unique(seg_slice).tolist())

# Display T2-FLAIR and segmentation
mri = image_slice[2].numpy()
seg = seg_slice.numpy()

masked_seg = np.ma.masked_where(seg == 0, seg)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.rot90(mri), cmap="gray")
plt.title("Normalized T2-FLAIR")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(mri), cmap="gray")
plt.imshow(
    np.rot90(masked_seg),
    cmap="jet",
    alpha=0.45,
    interpolation="nearest",
    vmin=0,
    vmax=3
)
plt.title("MRI + Tumor Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
