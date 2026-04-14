import sys
from pathlib import Path
import numpy as np
import json
import nibabel as nib
import pydicom

from mobstr3D.preprocessing.DICOM.index_inputs import check_DENSE_3D, find_dicom_for_nifti, find_phasedicom_for_nifti, get_slice_location, flag_slice_location, get_number_of_slices, get_number_of_frames, index_DENSE_series, collect_imaging_parameters
from mobstr3D.preprocessing.DICOM.segmentation.prepare_segmentation import prep_segmentation
from mobstr3D.preprocessing.DICOM.segmentation.process_segmentation import create_contours_from_labels
from mobstr3D.preprocessing.DICOM.phaseunwrapping.unwrap import extract_displacements
from mobstr3D.preprocessing.DICOM.transform_data import transform_to_pseudo_cardiac_coordinates



def perform_DICOM_preprocessing(config,mylogger):

    mylogger.info(f'Starting DICOM preprocessing...')
    
    # Define input path
    input_path = Path(config["preprocessing_paths"]["input_dir"])

    # Define output path
    output_path = Path(config["preprocessing_paths"]["output_dir"])

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)


    # 1. Index Inputs:

    # Check for existing index file
    index_filename = output_path / "DENSE_series_index.json"
    if not index_filename.exists():

        # Check inputs

        # Input files for DICOM preprocessing must be DENSE DICOM images

        if not input_path.exists():
            mylogger.error(f'Input path "{input_path}" does not exist.')
            sys.exit(1)

        input_files = list(input_path.glob("*.dcm"))
        if not input_files:
            mylogger.error(f'No input files found with ".dcm" in "{input_path}".')
            sys.exit(1)

        is_dense, file_index = check_DENSE_3D(input_files, mylogger)
        if not is_dense:
            mylogger.error(f'Some input files in "{input_path}" are not recognized as DENSE DICOM images. All input files must be DENSE DICOM images.')
            sys.exit(1)

        # Get number of frames and slices
        nSlices = get_number_of_slices(file_index, mylogger)
        nFrames = get_number_of_frames(file_index, mylogger)

        # Check number of input files (minimum: 3, warning: < 4)
        if nSlices < 3:
            mylogger.error(f'At least 3 input slices are required, but found {nSlices}.')
            sys.exit(1)
        elif nSlices < 4:
            mylogger.warning(f'Found {nSlices} input slices. Fewer than 4 can cause issues and require smoothing.')


        # Index into DENSE series

        DENSE_series_index = index_DENSE_series(file_index, nSlices, mylogger)

        index_filename = output_path / "DENSE_series_index.json"
        with open(index_filename, 'w') as f:
            json.dump(DENSE_series_index, f, indent=4, default=str)
        mylogger.info(f'Saved DENSE series index to "{index_filename}".')

        with open(index_filename, 'r') as f:
            DENSE_series_index = json.load(f)

    else:
        mylogger.info(f'Found existing DENSE series index at "{index_filename}". Loading...')
        with open(index_filename, 'r') as f:
            DENSE_series_index = json.load(f)

        nSlices = len(DENSE_series_index)
        iSlice = next(iter(DENSE_series_index))
        nSeries = len(DENSE_series_index[iSlice])
        iSeries = next(iter(DENSE_series_index[iSlice]))
        nFrames = len(DENSE_series_index[iSlice][iSeries])
        mylogger.info(f'Loaded DENSE series index with {nSlices} slices, {nSeries} series per slice, and {nFrames} frames per series.')

        # Check number of input files (minimum: 3, warning: < 4)
        if nSlices < 3:
            mylogger.error(f'At least 3 input slices are required, but found {nSlices}.')
            sys.exit(1)
        elif nSlices < 4:
            mylogger.warning(f'Found {nSlices} input slices. Fewer than 4 can cause issues and require smoothing.')


    # 2. Segmentation:

    # Prepare files for segmentation
    infer_path = prep_segmentation(config, DENSE_series_index, mylogger)

    # Apply chosen segmentation method to input files to extract contours
    if config["preprocessing_segmentation"]["segmentation_model"] == "modelESv2":
        from mobstr3D.preprocessing.DICOM.segmentation.segment_DENSE_modelESv2 import infer_labels
        model_path = Path("mobstr3D/preprocessing/DICOM/segmentation/modelESv2")
        infer_output = infer_labels(config, infer_path, model_path, mylogger)

    elif config["preprocessing_segmentation"]["segmentation_model"] == "modelallv2":
        mylogger.error(f'WIP: modelallv2 segmentation model not yet implemented.')
        sys.exit(1)
        from mobstr3D.preprocessing.DICOM.segmentation.segment_DENSE_modelallv2 import infer_labels
        model_path = Path("mobstr3D/preprocessing/DICOM/segmentation/modelallv2")
        infer_output = infer_labels(config, infer_path, model_path, mylogger)

    elif config["preprocessing_segmentation"]["segmentation_model"] == "custom":
        #INSERT CUSTOM SEGMENTATION METHOD HERE
        mylogger.error(f'Custom segmentation model not yet implemented.')
        sys.exit(1)
        
    else:
        mylogger.error(f'Invalid segmentation model: {config["preprocessing_segmentation"]["segmentation_model"]}')
        sys.exit(1)


    # 3. Contours:

    # Initialise dicts
    endo_contours_ics = {}
    epi_contours_ics = {}

    # Sort mask/labels by filename
    sorted_masks = sorted(list(Path(infer_output).glob("*.nii.gz")))

    # Load dicom-nifti key to match files
    dicom_nifti_key_path = Path(infer_path) / "dicom_nifti_key.json"
    # Load nifti_mask key to match files
    nifti_mask_key_path = Path(output_path) / "segmentation_output" / "nifti_mask_key.json"
    
    # Collect imaging parameters for scaling
    imaging_parameters = None  # To be filled after processing
    ip_flag = -1
    # Load the first slice and first frame phase dicom file (XPHA)
    dicom_data = pydicom.dcmread(str(DENSE_series_index['Slice_0']['XPHA']['0']))
    imaging_parameters = collect_imaging_parameters(ip_flag, imaging_parameters, dicom_data, mylogger)

    # Loop through labels to create contours from segmentation outputs
    for mask in sorted_masks:

        # Get slice index from mask filename "*_ID_'slice0''frame00'.nii.gz"
        slice_idx = int(mask.name.split("_")[3].split(".")[0][0])
        # Get frame index from mask filename "*_ID_'slice0''frame00'.nii.gz"
        frame_idx = int(mask.name.split("_")[3].split(".")[0][1:])

        # Ensure dicts have initialized dicts for this slice
        slice_key = f"Slice_{slice_idx}"
        if slice_key not in endo_contours_ics:
            endo_contours_ics[slice_key] = {}
            epi_contours_ics[slice_key] = {}

        # Create contours from segmentation files (labels)
        endo_contours_ics[slice_key][str(frame_idx)], epi_contours_ics[slice_key][str(frame_idx)] = create_contours_from_labels(DENSE_series_index, mask, imaging_parameters, config, mylogger)


    # 4. Displacements:
    
    # Initialise dicts
    locations_ics = {}
    displacements_ics = {}

    # Order input files based on slice_location (base to apex = +1 to -1)
    sorted_files = sorted(list(Path(infer_path).glob("*.nii.gz")))

    # Initialise dicoms dict
    sorted_dicoms = {}

    # Flag for inverted slice location
    flagInverted = flag_slice_location(pydicom.dcmread(str(find_dicom_for_nifti(dicom_nifti_key_path, sorted_files[0], mylogger))), mylogger)

    # Process inputs
    for nifti_idx, nifti in enumerate(sorted_files):

        mylogger.info(f'Processing input file: {nifti.name}')

        # # Get mask corresponding to current slice
        # mask = sorted_masks[nifti_idx]

        # Find dicom file corresponding to nifti file (MAG)
        dicom_mag = find_dicom_for_nifti(dicom_nifti_key_path, nifti, mylogger)
        # Find dicom phase file corresponding to nifti file (XPHA)
        dicom_phase = find_phasedicom_for_nifti(dicom_nifti_key_path, nifti, mylogger)
        
        # Load the dicom file (MAG)
        dicom_data_mag = pydicom.dcmread(str(dicom_mag))
        # Load the dicom file (XPHA)
        dicom_data_phase = pydicom.dcmread(str(dicom_phase))

        # a. Verify imaging parameters
        imaging_parameters = collect_imaging_parameters(nifti_idx, imaging_parameters, dicom_data_phase, mylogger)

        # b. Store sorted dicoms
        sorted_dicoms[nifti_idx] = dicom_mag

    # c. Extract displacements
    locations_ics, displacements_ics = extract_displacements(config, imaging_parameters, flagInverted, nSlices, nFrames, DENSE_series_index, sorted_dicoms, sorted_masks, mylogger)


    # 5. Transform outputs to pseudo-cardiac coordinate system

    # x = long-axis - out-of-plane
    # y and z = in-plane (Note: this could be improved by rotating with respect to the RV centroid)
    # origin = LV centroid

    # Transform the contours and displacements to the new coordinate system
    endo_contours_ccs, epi_contours_ccs, locations_ccs, displacements_ccs = transform_to_pseudo_cardiac_coordinates(config, imaging_parameters, endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics)
    mylogger.info('Transformed data to pseudo-cardiac coordinate system.')

    # Swap [slice] and [frame] dict order for ease of future processing
    # - from ["Slice_*"]["Frame"] to ["Frame"]["Slice_*"]
    def swap_dict_order(input_dict, imaging_parameters, config):
        nSlices = len(input_dict)
        frames = range(config["parameters"]["frame_of_seed"], imaging_parameters["Frames"])
        output_dict = {}
        for slice_idx in range(nSlices):
            slice_key = f"Slice_{slice_idx}"
            for frame_idx in frames:
                frame_key = str(frame_idx)
                if frame_key not in output_dict:
                    output_dict[frame_key] = {}
                output_dict[frame_key][slice_key] = input_dict[slice_key][frame_key]
        return output_dict

    endo_contours_ccs = swap_dict_order(endo_contours_ccs, imaging_parameters, config)
    epi_contours_ccs = swap_dict_order(epi_contours_ccs, imaging_parameters, config)
    locations_ccs = swap_dict_order(locations_ccs, imaging_parameters, config)
    displacements_ccs = swap_dict_order(displacements_ccs, imaging_parameters, config)


    # Save the processed data as structured .json files

    # Helper to convert numpy objects to JSON-serializable Python types
    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        return obj

    # Contours
    contours_filename = output_path / "contours.json"
    contours_dict = {
        "endo": endo_contours_ccs,
        "epi": epi_contours_ccs
    }
    with open(contours_filename, 'w') as f:
        json.dump(make_json_serializable(contours_dict), f, indent=4)

    # Displacements
    disp_filename = output_path / "displacements.json"
    disp_dict = {
        "locations": locations_ccs,
        "displacements": displacements_ccs
    }
    with open(disp_filename, 'w') as f:
        json.dump(make_json_serializable(disp_dict), f, indent=4)


    return
