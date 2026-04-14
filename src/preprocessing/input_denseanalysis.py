import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import json

from src.preprocessing.denseanalysis.extract_data import collect_imaging_parameters, get_slice_location, flag_slice_location, extract_contours, extract_displacements
from src.preprocessing.denseanalysis.transform_data import transform_to_pseudo_cardiac_coordinates



def perform_denseanalysis_preprocessing(config,mylogger):

    mylogger.info(f'Starting DENSEanalysis preprocessing...')
    
    # Define input path
    input_path = Path(config["preprocessing_paths"]["input_dir"])

    # Define output path
    output_path = Path(config["preprocessing_paths"]["output_dir"])

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)


    # Check inputs

    # Input files for DENSEanalysis preprocessing must be MATLAB output files from DENSEanalysis

    if not input_path.exists():
        mylogger.error(f'Input path "{input_path}" does not exist.')
        sys.exit(1)

    input_files = list(input_path.glob("*.mat"))
    if not input_files:
        mylogger.error(f'No input files found with ".mat" in "{input_path}".')
        sys.exit(1)

    # Check number of input files (minimum: 3, warning: < 4)
    if len(input_files) < 3:
        mylogger.error(f'At least 3 input files are required, but found {len(input_files)}.')
        sys.exit(1)
    elif len(input_files) < 4:
        mylogger.warning(f'Found {len(input_files)} input files. Fewer than 4 can cause issues and require smoothing.')


    # Predefine output dicts
    endo_contours_ics = {}
    epi_contours_ics = {}
    locations_ics = {}
    displacements_ics = {}
    imaging_parameters = None  # To be filled after processing the first file

    # Order input files based on slice_location (base to apex = +1 to -1)
    sorted_files = sorted(input_files, key=lambda f: get_slice_location(loadmat(f), mylogger), reverse=True)

    # Flag for inverted slice location
    flagInverted = flag_slice_location(loadmat(input_files[0]), mylogger)

    # Process inputs

    for slice_idx, input_file in enumerate(sorted_files):

        mylogger.info(f'Processing input file: {input_file}')
        
        # Load the MATLAB file
        mat_data = loadmat(input_file)

        # Ensure dicts have initialised dicts for this slice
        slice_key = f"Slice_{slice_idx}"
        if slice_key not in locations_ics:
            locations_ics[slice_key] = {}
            displacements_ics[slice_key] = {}

        # Step 1: Collect/check imaging parameters
        imaging_parameters = collect_imaging_parameters(slice_idx, imaging_parameters, mat_data, mylogger)

        # Step 2: Extract contours
        endo_contours_ics[slice_key], epi_contours_ics[slice_key] = extract_contours(config, imaging_parameters, mat_data, mylogger)

        # Step 3: Extract displacements
        locations_ics[slice_key], displacements_ics[slice_key] = extract_displacements(config, imaging_parameters, flagInverted, mat_data, mylogger)

    # Transform outputs to pseudo-cardiac coordinate system
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
