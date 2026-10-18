
import nibabel as nib
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.patients = sorted([
            folder for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        ])

        self.modalities = ["t1c", "t1n", "t2f", "t2w"]

    def __len__(self):
        return len(self.patients)

    def normalize(self, data):

        data = data.astype(np.float32)

        mask = data > 0

        normalized = np.zeros_like(data, dtype=np.float32)

        if np.any(mask):

            mean = np.mean(data[mask])
            std = np.std(data[mask])

            if std > 0:
                normalized[mask] = (data[mask] - mean) / std

        return normalized

    def __getitem__(self, index):

        patient_path = self.patients[index]
        patient_id = patient_path.name

        images = []

        for modality in self.modalities:

            filename = patient_path / f"{patient_id}-{modality}.nii.gz"

            image = nib.load(filename)
            data = image.get_fdata(dtype=np.float32)

            normalized = self.normalize(data)

            images.append(normalized)

        # Stack modalities into channel-first format
        image = np.stack(images, axis=0)

        # Load segmentation without normalization
        seg_file = patient_path / f"{patient_id}-seg.nii.gz"

        seg = np.asarray(nib.load(seg_file).dataobj)

        if not np.all(np.isfinite(seg)):
            raise ValueError(f"Invalid segmentation: {patient_id}")

        if not np.all(seg == np.floor(seg)):
            raise ValueError(f"Non-integer segmentation: {patient_id}")

        if not np.all(np.isin(seg, [0, 1, 2, 3])):
            raise ValueError(f"Unexpected segmentation labels: {patient_id}")

        if image.shape[1:] != seg.shape:
            raise ValueError(f"Shape mismatch: {patient_id}")

        seg = seg.astype(np.int64)

        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        seg_tensor = torch.from_numpy(seg.copy())

        return image_tensor, seg_tensor
