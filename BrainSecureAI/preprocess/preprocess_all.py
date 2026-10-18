
import nibabel as nib
import numpy as np
from pathlib import Path

DATASET_PATH = Path(
    r"D:\BrainSecureAI\dataset\BraTS\TrainingData"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_PATH = Path(r"D:\BrainSecureAI\outputs\preprocessed")

# Test only five patients first
MAX_PATIENTS = None

MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


def normalize_mri(data):
    data = data.astype(np.float32)
    mask = data > 0

    if not np.any(mask):
        raise ValueError("No positive MRI voxels found")

    mean = np.mean(data[mask])
    std = np.std(data[mask])

    normalized = np.zeros_like(data, dtype=np.float32)

    if std > 0:
        normalized[mask] = (data[mask] - mean) / std

    return normalized


def preprocess_patient(patient_path):
    patient_id = patient_path.name
        output_folder = OUTPUT_PATH / patient_id

    expected_files = [
        output_folder / f"{modality.upper()}_normalized.npy"
        for modality in MODALITIES
    ]
    expected_files.append(output_folder / "segmentation.npy")

    if all(file.exists() for file in expected_files):
        print(f"Skipping existing patient: {patient_id}")
        return
    print(f"\nProcessing: {patient_id}")

    processed = {}

    for modality in MODALITIES:
        file_path = patient_path / f"{patient_id}-{modality}.nii.gz"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        data = nib.load(file_path).get_fdata(dtype=np.float32)
        processed[modality] = normalize_mri(data)

    seg_path = patient_path / f"{patient_id}-seg.nii.gz"

    if not seg_path.exists():
        raise FileNotFoundError(seg_path)

    seg = np.asarray(nib.load(seg_path).dataobj)

    if not np.all(np.isfinite(seg)):
        raise ValueError("Non-finite segmentation values")

    if not np.all(seg == np.floor(seg)):
        raise ValueError("Non-integer segmentation labels")

    if np.min(seg) < 0 or np.max(seg) > 255:
        raise ValueError("Segmentation labels outside uint8 range")

    seg = seg.astype(np.uint8)

    for modality, data in processed.items():
        if data.shape != seg.shape:
            raise ValueError(f"Shape mismatch: {modality}")

    output_folder = OUTPUT_PATH / patient_id
    output_folder.mkdir(parents=True, exist_ok=True)

    for modality, data in processed.items():
        np.save(
            output_folder / f"{modality.upper()}_normalized.npy",
            data
        )

    np.save(output_folder / "segmentation.npy", seg)

    print("Shape:", seg.shape)
    print("Labels:", np.unique(seg))
    print("Status: SUCCESS")


def main():
    patients = sorted(
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )

    print("Total patient folders:", len(patients))
    print("Testing patients:", MAX_PATIENTS)

    successful = 0
    failed = 0

    for patient in patients[:MAX_PATIENTS] if MAX_PATIENTS is not None else patients:
        try:
            preprocess_patient(patient)
            successful += 1
        except Exception as error:
            failed += 1
            print(f"FAILED: {patient.name}")
            print("Reason:", error)

    print("\n========== SUMMARY ==========")
    print("Successful:", successful)
    print("Failed:", failed)
    print("=============================")


if __name__ == "__main__":
    main()
