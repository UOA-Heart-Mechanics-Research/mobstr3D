import os
import numpy as np
import nibabel as nib
import sys
from pathlib import Path
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set nnUNet environment variables so it doesn't scream at you with warnings
os.environ['nnUNet_raw'] = '.'
os.environ['nnUNet_preprocessed'] = '.'
os.environ['nnUNet_results'] = '.'

import nnunetv2 as nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


"""
    Apply nnU-Netv2 modelESv2 to prepared DENSE MAG images for segmentation.
    
    Note: This function assumes that the input images are already prepared and saved in NIfTI format

"""


def init_nnUNetv2(model_folder):
    # Check if GPU is available (torch)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = nnUNetPredictor(tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor



def plot_segmentation_over_image(seg_file, img_file):

    seg_nii = nib.load(str(seg_file))
    img_nii = nib.load(str(img_file))

    seg_data = seg_nii.get_fdata()
    img_data = img_nii.get_fdata()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_data, cmap='gray')
    ax[0].set_title('DENSE MAG Image')
    ax[1].imshow(img_data, cmap='gray')
    ax[1].imshow(seg_data, cmap='jet', alpha=0.15)
    # Create legend using proxy artists so labels appear
    handles = [
        mpatches.Patch(color='blue', label='0 = Background'),
        mpatches.Patch(color='green', label='1 = Myocardium'),
        mpatches.Patch(color='red', label='2 = Blood Pool'),
    ]
    ax[1].legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7)
    ax[1].set_title('Segmentation Overlay')
    plt.suptitle(f'{img_file.name}')
    plt.show()



def infer_labels(config, infer_path, model_path, mylogger):

    # Define model and paths
    model_folder_name = model_path / "nnUNetTrainer__nnUNetPlans__2d"

    input_folder = os.path.join(infer_path)
    output_folder = os.path.join(config["preprocessing_paths"]["output_dir"], "segmentation_output")

    # Only perform inference if labels/output folder doesn't already exist
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        predictor = init_nnUNetv2(model_folder_name)

        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

        mylogger.info(f'Segmentation inference complete. Segmentation files saved to "{output_folder}".')
    else:
        mylogger.info(f'Segmentation output folder "{output_folder}" already exists. Loading...')

    # Collect contours from segmentation outputs
    segmentation_files = list(Path(output_folder).glob("*.nii.gz"))
    if not segmentation_files:
        mylogger.error(f'No segmentation files found in "{output_folder}".')
        sys.exit(1)

    image_files = list(Path(input_folder).glob("*_0000.nii.gz"))
    if not image_files:
        mylogger.error(f'No input image files found in "{input_folder}".')
        sys.exit(1)

    # Sort files to ensure matching order
    segmentation_files.sort()
    image_files.sort()

    # Define nifti-mask key
    nifti_mask_key = []
    for seg_file, img_file in zip(segmentation_files, image_files):

        # Check matching filenames (slice and frame in string)
        seg_index = seg_file.name.split(".")[0].split("_")[3]
        img_index = img_file.name.split(".")[0].split("_")[3]
        if seg_index != img_index:
            mylogger.error(f'Segmentation file "{seg_file.name}" does not match image file "{img_file.name}".')
            sys.exit(1)

        nifti_mask_key.append({
            "slice_index": seg_index[0],
            "frame_index": seg_index[1:],
            "nifti_file": img_file,
            "mask_file": seg_file
        })


    # Debug: plot segmentations over images
    if config["debug_flags"]["debug_segmentation"]:
        mylogger.info(f'Plotting segmentation overlays for debugging...')

        for seg_file, img_file in zip(segmentation_files, image_files):

            # Check matching filenames (slice and frame in string)
            seg_index = seg_file.name.split(".")[0].split("_")[3]
            img_index = img_file.name.split(".")[0].split("_")[3]
            if seg_index != img_index:
                mylogger.warning(f'Segmentation file "{seg_file.name}" does not match image file "{img_file.name}". Skipping visualization.')
                continue
            
            plot_segmentation_over_image(seg_file, img_file)

    # Save nifti-mask key for reference
    nifti_mask_key_path = Path(output_folder) / "nifti_mask_key.json"
    with open(nifti_mask_key_path, "w") as f:
        json.dump(nifti_mask_key, f, indent=4, default=str)
    mylogger.info(f'Saved NIfTI-mask key to "{nifti_mask_key_path}".')

    return output_folder