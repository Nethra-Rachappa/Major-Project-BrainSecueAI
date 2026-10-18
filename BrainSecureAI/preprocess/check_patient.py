import nibabel as nib
import numpy as np
from pathlib import Path

patient_path = Path(
    r"D:\BrainSecureAI\dataset\BraTS\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    r"\BraTS-GLI-01535-000"
)

files = {
    "T1c": "BraTS-GLI-01535-000-t1c.nii.gz",
    "T1n": "BraTS-GLI-01535-000-t1n.nii.gz",
    "T2f": "BraTS-GLI-01535-000-t2f.nii.gz",
    "T2w": "BraTS-GLI-01535-000-t2w.nii.gz",
    "Segmentation": "BraTS-GLI-01535-000-seg.nii.gz"
}

for name, filename in files.items():

    file_path = patient_path / filename

    image = nib.load(file_path)
    data = image.get_fdata()

    print("\n", name)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))