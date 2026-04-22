import sys
import numpy as np
from pathlib import Path
import pydicom
import nibabel as nib
import os
import json


"""
Module to prepare DICOM data for segmentation.

Input: DENSE_series_index (dict) - Indexed DENSE series information from input DICOM files
    slice_#: {
            MAG: {
                frame_#: file_path
            }
    }

Sets up nifti files for ready for inference.
MAG channel only.

"""

def load_dicom_img(file_path, mylogger):
    """
    Load a DICOM image from the given file path.
    """
    try:
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        caseID = getattr(ds, "PatientID", "UnknownPatient")
        if caseID == "":
            caseID = "UnknownPatient"
        mylogger.info(f'Loaded DICOM image from "{file_path}".')
        return img, caseID
    except Exception as e:
        mylogger.error(f'Error loading DICOM image from "{file_path}": {e}')
        raise


def prep_segmentation(config, DENSE_series_index, mylogger):

    # Define output directory for segmentation inference files
    output_dir = Path(config["preprocessing_paths"]["output_dir"]) / "segmentation_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frames to process based on config
    nFrames = len(next(iter(next(iter(DENSE_series_index.values())).values())))
    frames = range(config["parameters"]["frame_of_seed"], nFrames)

    # Setup dicom-nifti key
    dicom_nifti_key = []

    # Loop through slices
    for slice_idx, slice_key in enumerate(DENSE_series_index):

        # Loop through frames
        for frame_idx in frames:

            mylogger.info(f'Fitting frame {frame_idx} of slice {slice_idx}...')

            # Load DICOM image for MAG channel
            dicom_file = DENSE_series_index[slice_key]["MAG"][str(frame_idx)]
            dicom_file_phase = DENSE_series_index[slice_key]["XPHA"][str(frame_idx)]
            dicom_img, caseID = load_dicom_img(dicom_file, mylogger)

            # Save nifty file for future segmentation inference
            affine = np.eye(4)
            nii_img = nib.Nifti1Image(dicom_img, affine)
            # Format naming for nnU-Netv2 - where the first index is slice_id and last is frame_number
            nifti_filename = f"{caseID}_{slice_idx:02d}{frame_idx:02d}"

            nib.save(nii_img, os.path.join(output_dir, f"{nifti_filename}_0000.nii.gz"))

            dicom_nifti_key.append({
                "slice_index": slice_idx,
                "frame_index": frame_idx,
                "dicom_file": dicom_file,
                "dicom_file_phase": dicom_file_phase,
                "nifti_file": output_dir / f"{nifti_filename}_0000.nii.gz"
            })

    # Save dicom-nifti key for reference
    dicom_nifti_key_path = Path(output_dir) / "dicom_nifti_key.json"
    with open(dicom_nifti_key_path, "w") as f:
        json.dump(dicom_nifti_key, f, indent=4, default=str)
    mylogger.info(f'Saved DICOM-NIfTI key to "{dicom_nifti_key_path}".')

    mylogger.info(f'Segmentation preparation complete. Nifti files saved to "{output_dir}".')
    
    return output_dir
